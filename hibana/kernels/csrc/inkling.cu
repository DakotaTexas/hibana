// Inkling 2 decode kernels: what a layer visit of a few rows (one a request, or a request's verified rows) runs
// between its experts' products (`inkling_prefill.py`, on the tensor cores). Each serves a captured decode graph:
// fixed grids, every size from its tensors. A visit is bound by memory traffic, not arithmetic, and one block draws
// only about 5 GB/s of cold weights (one SM's share of the memory system), so whatever reads more than a few tens of
// KB spreads over many blocks.
//
// Attention (`attention`): block (rows of a sequence, KV head, split) scores its share of the rows' keys for the
//   head's three query heads in chunks with a running maximum (online softmax), then `merge` joins the splits with
//   the sink logit and applies the output gate. A row is (sequence, visible keys): key j of a sequence is the token at position j, so
//   the last visible key is the query's own and the bias index is their distance.
//
// The steps between (`mix` and the gate kernels, `rms`, `conv`, `prep`, `route`), the projection (`gemv`) and the
// pass after the experts' products (`combine`) are described where they are defined.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr float kNegInf = -1e30f;

__device__ __forceinline__ float bf16_to_float(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

__device__ __forceinline__ uint16_t float_to_bf16(float f) {
  const uint32_t u = __float_as_uint(f);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);  // round to nearest even
}

__device__ __forceinline__ float round_bf16(float v) { return bf16_to_float(float_to_bf16(v)); }

// the four E4M3 values of a word, low byte first, as floats: through the hardware's (exact) E4M3 -> FP16 conversion
__device__ __forceinline__ void e4m3x4_to_float(uint32_t w, float* f) {
  uint32_t lo, hi;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"(static_cast<uint16_t>(w & 0xffffu)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"(static_cast<uint16_t>(w >> 16)));
  asm("{.reg .b16 a, b; mov.b32 {a, b}, %2; cvt.f32.f16 %0, a; cvt.f32.f16 %1, b;}" : "=f"(f[0]), "=f"(f[1]) : "r"(lo));
  asm("{.reg .b16 a, b; mov.b32 {a, b}, %2; cvt.f32.f16 %0, a; cvt.f32.f16 %1, b;}" : "=f"(f[2]), "=f"(f[3]) : "r"(hi));
}

// ---------------------------------------------------------------------------------------------------- attention

constexpr int kHeadDim = 128;
constexpr int kGroup = 3;  // query heads per KV head
constexpr int kAttnThreads = kGroup * kHeadDim;
constexpr int kRelMax = 16;     // the relative-position states' width
constexpr int kSplitKeys = 32;  // fewest keys a split takes: a short row leaves its later blocks empty
constexpr int kPartial = 2 + kHeadDim;  // a split's running maximum, weight sum and weighted values, per query head

// grid (rows / group * kv_heads, splits): a block a share of the keys of `group` consecutive rows of one sequence
// (decode: 1; verification: the pending token and its drafts) for the KV head's three query heads, in chunks of 96.
// Rows of a sequence read the same pages, so the block loads and converts each key once for all its rows (R >= group,
// rows past `group` idle) and masks a key a row does not see: four rows over 32k cold keys took 1,323 us as a block a
// row (3.4x one row), 428 us so. Four threads a key each read a quarter of its K and V rows and of the rows' bias
// rows in one run of loads, issued a chunk ahead (the loads of chunk c + 1 are in flight while chunk c is scored and
// summed: one row over 100k keys 1,206 -> 1,070 us), score it against the staged queries and leave the V row in shared
// memory by dimension; the block's maxima join the running ones (online softmax); then a thread a (head, dimension)
// sums the weighted values of every row from shared memory, sixteen bytes a load. A block's time is its longest
// thread's instructions, so both steps divide evenly among the threads (a thread a key scoring all 128 dimensions
// held a short row to 4 us). A key at backward distance d < extent adds the bias BF16(rel . proj[d]) to its score
// (rel [rows, heads, 16] FP32, proj [extent, 16] BF16 or null).
// Two- and four-row blocks are held to two a multiprocessor (80 registers; four rows at 32k keys 909 -> 869 us).
template <bool kFp8, int R>
__global__ void __launch_bounds__(kAttnThreads, R == 2 || R == 4 ? 2 : 1) attention_kernel(
    const uint16_t* __restrict__ q, const float* __restrict__ rel, const uint16_t* __restrict__ proj, const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool, const int32_t* __restrict__ indptr, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ row_seq, const int32_t* __restrict__ row_len, float* __restrict__ partial,
    int kv_heads, int extent, int window, int splits, int group, float scale) {
  constexpr int elem = kFp8 ? 1 : 2;
  constexpr int parts = 4;                       // threads a key
  constexpr int chunk = kAttnThreads / parts;    // keys staged at a time
  constexpr int dims = kHeadDim / parts;         // dimensions a thread scores
  constexpr int run = 16 / elem;                 // values a 16-byte load holds
  constexpr int loads = dims / run;              // 16-byte loads a thread's share of a K or V row
  constexpr int warps = kAttnThreads / 32;
  constexpr int RH = R * kGroup;                 // (row, query head) pairs a block scores
  static_assert(RH * kRelMax <= kAttnThreads, "a thread loads one relative-state value");
  constexpr int qpad = dims + 4;                 // a quarter's stride in qs: the four quarters' loads hit different banks
  __shared__ __align__(16) float qs[RH * parts * qpad];
  __shared__ float relq[RH * kRelMax];
  __shared__ float part[warps * RH];
  __shared__ float stat[2 * RH];                                     // the running maxima; what the sums so far shrink by
  __shared__ __align__(16) float weight[RH * chunk];                 // a chunk's softmax weights
  // its V rows, [dimension][key]; a quarter q of the dimensions keeps key j at column j ^ 8q, else the four quarters'
  // stores of one key land in one bank (rows 32 apart). With the queries' padded stride and E4M3 converted by the
  // hardware instead of a shared table (random indices, bank conflicts), four rows at 32k keys took 870 -> 460 us
  __shared__ __align__(16) uint8_t staged[kHeadDim * chunk * elem];

  const int g = blockIdx.x % kv_heads, split = blockIdx.y;
  const int row0 = (blockIdx.x / kv_heads) * group;
  const int heads = kv_heads * kGroup;
  const int t = threadIdx.x, th = t / kHeadDim, td = t % kHeadDim;
  // every load that needs no other first, so their trips overlap (a warp stalls at the first use of a pending load
  // and issues nothing behind it): the rows' lengths, sequence, queries and relative states
  const int seq = row_seq[row0];
  int n[R], lo[R];  // a row sees keys lo .. n - 1; an idle row none
  float qv[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    n[r] = r < group ? row_len[row0 + r] : 0;
    qv[r] = r < group ? bf16_to_float(q[(static_cast<int64_t>(row0 + r) * heads + g * kGroup) * kHeadDim + t]) : 0.f;
  }
  const int rr = t / (kGroup * kRelMax);
  const float relv = proj != nullptr && t < RH * kRelMax && rr < group
                         ? rel[(static_cast<int64_t>(row0 + rr) * heads + g * kGroup) * kRelMax + t % (kGroup * kRelMax)] : 0.f;
  const int32_t* pages = indices + indptr[seq];
  int lo_all = 0x7fffffff, hi_all = 0;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    lo[r] = window > 0 && n[r] > window ? n[r] - window : 0;
    if (r < group) {
      lo_all = min(lo_all, lo[r]);
      hi_all = max(hi_all, n[r]);
    }
  }
  const int per = max((hi_all - lo_all + splits - 1) / splits, kSplitKeys);
  const int first = lo_all + split * per, last = min(hi_all, first + per);
  auto at = [&](int r, int h) { return partial + ((static_cast<int64_t>(row0 + r) * heads + g * kGroup + h) * splits + split) * kPartial; };
  if (first >= last) {
    if (td == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (r < group) {
          at(r, th)[0] = kNegInf;
          at(r, th)[1] = 0.f;
        }
      }
    }
    return;
  }
  const int64_t page_bytes = static_cast<int64_t>(kv_heads) * kHeadDim * elem;
  const int j = t / parts, s = t % parts;  // scoring: this thread's key of the chunk and quarter of its dimensions
  const int64_t off = (static_cast<int64_t>(g) * kHeadDim + s * dims) * elem;
  uint4 kw[loads], vw[loads];  // the K and V words of this thread's key in the next chunk
  uint2 pw[R];                 // and a quarter of its bias row for each row: four BF16
  auto fetch = [&](int key, int page) {
    const uint4* kp = reinterpret_cast<const uint4*>(k_pool + page * page_bytes + off);
    const uint4* vp = reinterpret_cast<const uint4*>(v_pool + page * page_bytes + off);
#pragma unroll
    for (int i = 0; i < loads; ++i) {
      kw[i] = kp[i];
      vw[i] = vp[i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int dist = n[r] - 1 - key;
      pw[r] = proj != nullptr && key >= lo[r] && dist >= 0 && dist < extent
                  ? __ldg(reinterpret_cast<const uint2*>(proj + static_cast<int64_t>(dist) * kRelMax + s * (kRelMax / parts)))
                  : make_uint2(0u, 0u);
    }
  };
  const int page0 = first + j < last ? pages[first + j] : 0;
  int ahead = first + chunk + j < last ? pages[first + chunk + j] : 0;  // the page of this thread's key a chunk on
#pragma unroll
  for (int r = 0; r < R; ++r) {
    qs[(r * kGroup + th) * parts * qpad + (td / dims) * qpad + td % dims] = qv[r];
  }
  if (t < RH * kRelMax) {
    relq[t] = relv;
  }
  if (first + j < last) {
    fetch(first + j, page0);
  }
  float top = kNegInf;                  // threads 0 .. RH - 1: their (row, head)'s running maximum
  float acc[R], sum[R];                 // this thread's (row, head, dimension) sums; the (row, head)'s weight sums
#pragma unroll
  for (int r = 0; r < R; ++r) {
    acc[r] = 0.f;
    sum[r] = 0.f;
  }
  for (int base = first; base < last; base += chunk) {
    const int count = min(chunk, last - base), key = base + j;
    __syncthreads();
    float dot[RH], bias[RH];
#pragma unroll
    for (int x = 0; x < RH; ++x) {
      dot[x] = 0.f;
      bias[x] = 0.f;
    }
    if (j < count) {
#pragma unroll
      for (int i = 0; i < loads; ++i) {
        const uint32_t kword[4] = {kw[i].x, kw[i].y, kw[i].z, kw[i].w}, vword[4] = {vw[i].x, vw[i].y, vw[i].z, vw[i].w};
        float kv[run];
#pragma unroll
        for (int b = 0; b < run; ++b) {
          const int d = s * dims + i * run + b;
          if constexpr (kFp8) {
            if (b % 4 == 0) {
              e4m3x4_to_float(kword[b / 4], kv + b);
            }
            staged[d * chunk + (j ^ (8 * s))] = static_cast<uint8_t>(vword[b / 4] >> (8 * (b % 4)));
          } else {
            kv[b] = bf16_to_float(static_cast<uint16_t>(kword[b / 2] >> (16 * (b % 2))));
            reinterpret_cast<uint16_t*>(staged)[d * chunk + (j ^ (8 * s))] = static_cast<uint16_t>(vword[b / 2] >> (16 * (b % 2)));
          }
        }
#pragma unroll
        for (int x = 0; x < RH; ++x) {
          float qv[run];
#pragma unroll
          for (int b = 0; b < run; b += 4) {
            const float4 q4 = *reinterpret_cast<const float4*>(qs + x * parts * qpad + s * qpad + i * run + b);
            qv[b] = q4.x;
            qv[b + 1] = q4.y;
            qv[b + 2] = q4.z;
            qv[b + 3] = q4.w;
          }
#pragma unroll
          for (int b = 0; b < run; ++b) {
            dot[x] = fmaf(qv[b], kv[b], dot[x]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t word[2] = {pw[r].x, pw[r].y};
#pragma unroll
        for (int d = 0; d < kRelMax / parts; ++d) {
          const float pv = bf16_to_float(static_cast<uint16_t>(word[d / 2] >> (16 * (d % 2))));
#pragma unroll
          for (int h = 0; h < kGroup; ++h) {
            bias[r * kGroup + h] = fmaf(relq[(r * kGroup + h) * kRelMax + s * (kRelMax / parts) + d], pv, bias[r * kGroup + h]);
          }
        }
      }
    }
    if (key + chunk < last) {  // the next chunk's loads fly while this one is reduced and summed
      fetch(key + chunk, ahead);
    }
    ahead = key + 2 * chunk < last ? pages[key + 2 * chunk] : 0;
    float score[RH];
#pragma unroll
    for (int x = 0; x < RH; ++x) {  // a key's four threads join their quarters; then the chunk's maximum
      const int r = x / kGroup;
      dot[x] += __shfl_xor_sync(0xffffffffu, dot[x], 1);
      dot[x] += __shfl_xor_sync(0xffffffffu, dot[x], 2);
      bias[x] += __shfl_xor_sync(0xffffffffu, bias[x], 1);
      bias[x] += __shfl_xor_sync(0xffffffffu, bias[x], 2);
      score[x] = j < count && key >= lo[r] && key < n[r] ? dot[x] * scale + round_bf16(bias[x]) : kNegInf;
      float best = score[x];
      for (int o = parts; o < 32; o <<= 1) {  // a key's four threads hold the same score
        best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
      }
      if ((t & 31) == 0) {
        part[(t >> 5) * RH + x] = best;
      }
    }
    __syncthreads();
    if (t < RH) {
      float most = top;
#pragma unroll
      for (int w = 0; w < warps; ++w) {
        most = fmaxf(most, part[w * RH + t]);
      }
      stat[t] = most;
      stat[RH + t] = expf(top - most);
      top = most;
    }
    __syncthreads();
    if (j < count) {  // a key's four threads share its (row, head) weights; a key a row does not see weighs 0
#pragma unroll
      for (int x = 0; x < RH; ++x) {
        if (x % parts == s) {
          weight[x * chunk + j] = score[x] > kNegInf ? expf(score[x] - stat[x]) : 0.f;
        }
      }
    }
    float shrink[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      shrink[r] = stat[RH + r * kGroup + th];
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] *= shrink[r];
      sum[r] *= shrink[r];
    }
    for (int k = 0; k < count; k += run) {
      const int sw = 8 * (td / dims);  // the quarter's swizzle: 16-byte blocks move, and FP8's 8-key halves swap
      const uint4 v = *reinterpret_cast<const uint4*>(staged + (td * chunk + (k ^ (sw & ~(run - 1)))) * elem);
      const bool swap = kFp8 && (sw & 8);
      const uint32_t word[4] = {swap ? v.z : v.x, swap ? v.w : v.y, swap ? v.x : v.z, swap ? v.y : v.w};
      float vv[run];
#pragma unroll
      for (int u = 0; u < run; ++u) {
        if constexpr (kFp8) {
          if (u % 4 == 0) {
            e4m3x4_to_float(word[u / 4], vv + u);
          }
        } else {
          vv[u] = bf16_to_float(static_cast<uint16_t>(word[u / 2] >> (16 * (u % 2))));
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float wv[run];
#pragma unroll
        for (int u = 0; u < run; u += 4) {
          const float4 w4 = *reinterpret_cast<const float4*>(weight + (r * kGroup + th) * chunk + k + u);
          wv[u] = w4.x;
          wv[u + 1] = w4.y;
          wv[u + 2] = w4.z;
          wv[u + 3] = w4.w;
        }
#pragma unroll
        for (int u = 0; u < run; ++u) {
          if (k + u < count) {
            acc[r] = fmaf(wv[u], vv[u], acc[r]);
            sum[r] += wv[u];
          }
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r < group) {
      at(r, th)[2 + td] = acc[r];
      if (td == 0) {
        at(r, th)[1] = sum[r];
      }
    }
  }
  if (t < RH && t / kGroup < group) {
    at(t / kGroup, t % kGroup)[0] = top;
  }
}

constexpr int kMaxSplits = kHeadDim;  // the merge reads a split's statistics a thread

// grid (rows * heads): join the splits and the sink, and apply the output gate sigmoid(gate) when given (`gate`: the
// row's BF16 gate values, `gate_stride` elements a row). Thread s reads split s's maximum and weight sum, so the
// splits' scales come from one parallel round of loads and one block reduction rather than a loop of dependent ones.
__global__ void __launch_bounds__(kHeadDim) attention_merge_kernel(const float* __restrict__ partial,
                                                                  const float* __restrict__ sink,
                                                                  const uint16_t* __restrict__ gate, int64_t gate_stride,
                                                                  uint16_t* __restrict__ out, int splits, int heads) {
  __shared__ float scale_of[kMaxSplits];
  __shared__ float red[2 * (kHeadDim / 32)];
  const float* mine = partial + static_cast<int64_t>(blockIdx.x) * splits * kPartial;
  const int t = threadIdx.x;
  float gv = 0.f;
  if (gate != nullptr) {
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    gv = bf16_to_float(gate[row * gate_stride + head * kHeadDim + t]);
  }
  const float sk = sink[blockIdx.x];
  float top_s = kNegInf, sum_s = 0.f;
  if (t < splits) {  // every split writes both, an empty one (-inf, 0)
    const float raw = mine[t * kPartial];
    sum_s = mine[t * kPartial + 1];
    top_s = sum_s > 0.f ? raw : kNegInf;
  }
  float most = top_s;
  for (int o = 16; o > 0; o >>= 1) {
    most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, o));
  }
  if ((t & 31) == 0) {
    red[t >> 5] = most;
  }
  __syncthreads();
  float top = sk;
#pragma unroll
  for (int w = 0; w < kHeadDim / 32; ++w) {
    top = fmaxf(top, red[w]);
  }
  const float ws = sum_s > 0.f ? expf(top_s - top) : 0.f;
  if (t < splits) {
    scale_of[t] = ws;
  }
  float part = ws * sum_s;
  for (int o = 16; o > 0; o >>= 1) {
    part += __shfl_xor_sync(0xffffffffu, part, o);
  }
  if ((t & 31) == 0) {
    red[kHeadDim / 32 + (t >> 5)] = part;
  }
  __syncthreads();
  float total = expf(sk - top);
#pragma unroll
  for (int w = 0; w < kHeadDim / 32; ++w) {
    total += red[kHeadDim / 32 + w];
  }
  float acc = 0.f;
  int s = 0;
  for (; s + 4 <= splits; s += 4) {  // four independent loads a round; an empty split's values are never read (unwritten)
    float w[4], pv[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      w[u] = scale_of[s + u];
      pv[u] = w[u] > 0.f ? mine[(s + u) * kPartial + 2 + t] : 0.f;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      acc = fmaf(w[u], pv[u], acc);
    }
  }
  for (; s < splits; ++s) {
    const float w = scale_of[s];
    acc = w > 0.f ? fmaf(w, mine[s * kPartial + 2 + t], acc) : acc;
  }
  float value = acc / total;
  if (gate != nullptr) {
    value = value / (1.f + expf(-gv));
  }
  out[static_cast<int64_t>(blockIdx.x) * kHeadDim + t] = float_to_bf16(value);
}

// ---------------------------------------------------------------------------------------------------- decode glue
//
// The small steps between a decode row's projections, one launch each instead of some two hundred PyTorch operators a
// layer visit: `mix` (a sublayer input: the block attention residual, the message gate, the RMS norm), `conv` (a
// hidden-stream convolution with its residual), `prep` (queries, relative states, keys and values from the three
// projections, K/V written to their page) and `route` (the experts' router).

constexpr int kMixThreads = 256;
constexpr int kMixPer = 9;  // values a thread holds: a row of 2,304 over the block
constexpr float kEps = 1e-6f;

// The glue kernels below run one block a row, and a block's time is its threads' chains of loads, not their
// arithmetic: a load whose value the next instruction uses stalls its warp for the trip to the cache, so every
// kernel reads what it needs in runs of independent loads into registers (loop bounds known at compile time) and
// computes afterwards. Loading and accumulating value by value, the mix took 13 us an output and a short row's
// attention 9; in runs, 2 to 3.

// The sum of every thread's `v`, returned to all of them. scratch: one float a warp.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  if ((threadIdx.x & 31) == 0) {
    scratch[threadIdx.x >> 5] = v;
  }
  __syncthreads();
  float total = 0.f;
  for (int w = 0; w < static_cast<int>(blockDim.x) / 32; ++w) {
    total += scratch[w];
  }
  __syncthreads();
  return total;
}

// The sums of every thread's v[0 .. N) over a block of kMixThreads, in place. scratch: [warps][N], which no other
// reduction of the kernel uses (so one barrier serves).
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float* scratch) {
#pragma unroll
  for (int n = 0; n < N; ++n) {
    for (int off = 16; off > 0; off >>= 1) {
      v[n] += __shfl_xor_sync(0xffffffffu, v[n], off);
    }
    if ((threadIdx.x & 31) == 0) {
      scratch[(threadIdx.x >> 5) * N + n] = v[n];
    }
  }
  __syncthreads();
  float part[kMixThreads / 32][N];
#pragma unroll
  for (int w = 0; w < kMixThreads / 32; ++w) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
      part[w][n] = scratch[w * N + n];
    }
  }
#pragma unroll
  for (int n = 0; n < N; ++n) {
    v[n] = 0.f;
#pragma unroll
    for (int w = 0; w < kMixThreads / 32; ++w) {
      v[n] += part[w][n];
    }
  }
}

// grid (rows, outputs), a block an output, over the row's kCount normalized states and rms(partial):
//   m = sum_n softmax_n(S_n . query / sqrt(H)) S_n
//   out = BF16(rms(m) * weight);
//   or, for a gated input (`gated` given): m, the message msg (the same mix of the row's two message sources by
//   message query g0 + j) and `pair` = BF16 [rms(m) | rms(msg)], for the gate kernels below.
// states [rows, slots, H] FP32; partial [rows, H] BF16 or FP32; queries [Q, H] (output j reads min(j, Q - 1));
// sources [rows, 2, H], message queries [G, H]; gated, message [outputs, rows, H]; pair [outputs, rows, 2 H].
template <bool kPartialBf16, int kCount>
__global__ void __launch_bounds__(kMixThreads) mix_kernel(
    const float* __restrict__ states, int slots, const void* __restrict__ partial, const float* __restrict__ queries, int nq,
    const float* __restrict__ sources, const float* __restrict__ msg_queries, int g0, float* __restrict__ gated,
    float* __restrict__ message, uint16_t* __restrict__ pair, const float* __restrict__ weight, uint16_t* __restrict__ out) {
  constexpr int hidden = kMixThreads * kMixPer, terms = kCount + 4;
  __shared__ float scratch[kMixThreads / 32 * terms];
  __shared__ float scratch2[kMixThreads / 32 * 2];
  const int row = blockIdx.x, j = blockIdx.y, t = threadIdx.x, rows = gridDim.x;
  const bool merge = gated != nullptr;
  const float* S = states + static_cast<int64_t>(row) * slots * hidden + t;
  const float* q = queries + static_cast<int64_t>(min(j, nq - 1)) * hidden + t;
  const int64_t own_at = static_cast<int64_t>(row) * hidden + t;
  float own[kMixPer], qv[kMixPer], s[kCount + 1][kMixPer], a[kMixPer], b[kMixPer], mq[kMixPer];
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    if constexpr (kPartialBf16) {
      own[i] = bf16_to_float(reinterpret_cast<const uint16_t*>(partial)[own_at + i * kMixThreads]);
    } else {
      own[i] = reinterpret_cast<const float*>(partial)[own_at + i * kMixThreads];
    }
    qv[i] = q[i * kMixThreads];
#pragma unroll
    for (int n = 0; n < kCount; ++n) {
      s[n][i] = S[n * hidden + i * kMixThreads];
    }
    a[i] = b[i] = mq[i] = 0.f;
  }
  if (merge) {
    const float* src = sources + static_cast<int64_t>(row) * 2 * hidden + t;
    const float* mqp = msg_queries + static_cast<int64_t>(g0 + j) * hidden + t;
#pragma unroll
    for (int i = 0; i < kMixPer; ++i) {
      a[i] = src[i * kMixThreads];
      b[i] = src[hidden + i * kMixThreads];
      mq[i] = mqp[i * kMixThreads];
    }
  }
  float sums[terms];  // own . own, own . query, the message sources . message query, then each state . query
#pragma unroll
  for (int n = 0; n < terms; ++n) {
    sums[n] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    sums[0] = fmaf(own[i], own[i], sums[0]);
    sums[1] = fmaf(own[i], qv[i], sums[1]);
    sums[2] = fmaf(a[i], mq[i], sums[2]);
    sums[3] = fmaf(b[i], mq[i], sums[3]);
#pragma unroll
    for (int n = 0; n < kCount; ++n) {
      sums[4 + n] = fmaf(s[n][i], qv[i], sums[4 + n]);
    }
  }
  block_sums(sums, scratch);
  const float inv_h = 1.f / static_cast<float>(hidden), root = rsqrtf(static_cast<float>(hidden));
  const float own_scale = rsqrtf(sums[0] * inv_h + kEps);
  float share[kCount + 1];
  float top = share[kCount] = sums[1] * own_scale * root, total = 0.f;
#pragma unroll
  for (int n = 0; n < kCount; ++n) {
    share[n] = sums[4 + n] * root;
    top = fmaxf(top, share[n]);
  }
#pragma unroll
  for (int n = 0; n <= kCount; ++n) {
    share[n] = expf(share[n] - top);
    total += share[n];
  }
  const float wa = 1.f / (1.f + expf((sums[3] - sums[2]) * root)), wb = 1.f - wa;
  float m[kMixPer], squares[2] = {0.f, 0.f};
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    float v = share[kCount] * own[i] * own_scale;
#pragma unroll
    for (int n = 0; n < kCount; ++n) {
      v = fmaf(share[n], s[n][i], v);
    }
    m[i] = v / total;
    a[i] = wa * a[i] + wb * b[i];  // the message
    squares[0] = fmaf(m[i], m[i], squares[0]);
    squares[1] = fmaf(a[i], a[i], squares[1]);
  }
  block_sums(squares, scratch2);
  const float scale = rsqrtf(squares[0] * inv_h + kEps);
  const int64_t at = (static_cast<int64_t>(j) * rows + row) * hidden + t;  // [outputs, rows, H]
  if (merge) {  // the gate kernels take it from here
    const float msg_scale = rsqrtf(squares[1] * inv_h + kEps);
#pragma unroll
    for (int i = 0; i < kMixPer; ++i) {
      gated[at + i * kMixThreads] = m[i];
      message[at + i * kMixThreads] = a[i];
      pair[2 * at - t + i * kMixThreads] = float_to_bf16(m[i] * scale);
      pair[2 * at - t + hidden + i * kMixThreads] = float_to_bf16(a[i] * msg_scale);
    }
    return;
  }
  const float* w = weight + static_cast<int64_t>(j) * hidden + t;
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    qv[i] = w[i * kMixThreads];
  }
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    out[at + i * kMixThreads] = float_to_bf16(m[i] * scale * qv[i]);
  }
}

// grid (rows): out = x rsqrt(mean(x^2) + eps) (* weight) over rows of `width` <= 2,304, as BF16 or FP32, row r written
// at out + r * out_stride elements.
template <bool kInBf16, bool kOutBf16>
__global__ void __launch_bounds__(kMixThreads) rms_kernel(const void* __restrict__ x, const float* __restrict__ weight,
                                                          void* __restrict__ out, int64_t out_stride, int width, float eps) {
  __shared__ float scratch[kMixThreads / 32];
  const int64_t row = blockIdx.x;
  const int t = threadIdx.x;
  float v[kMixPer], w[kMixPer];
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    const int h = t + i * kMixThreads;
    v[i] = 0.f;
    w[i] = 1.f;
    if (h < width) {
      if constexpr (kInBf16) {
        v[i] = bf16_to_float(reinterpret_cast<const uint16_t*>(x)[row * width + h]);
      } else {
        v[i] = reinterpret_cast<const float*>(x)[row * width + h];
      }
      if (weight != nullptr) {
        w[i] = weight[h];
      }
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    ss = fmaf(v[i], v[i], ss);
  }
  const float scale = rsqrtf(block_sum(ss, scratch) / static_cast<float>(width) + eps);
#pragma unroll
  for (int i = 0; i < kMixPer; ++i) {
    const int h = t + i * kMixThreads;
    if (h < width) {
      const float value = weight != nullptr ? v[i] * scale * w[i] : v[i] * scale;
      if constexpr (kOutBf16) {
        reinterpret_cast<uint16_t*>(out)[row * out_stride + h] = float_to_bf16(value);
      } else {
        reinterpret_cast<float*>(out)[row * out_stride + h] = value;
      }
    }
  }
}
// The message gate, m <- (1 - tanh(forget low)) m + tanh(gain low) msg with low = down [rms(m); rms(msg)], then
// rms(m) * weight. One block draws about 5 GB/s of cold weights (one SM's share of the memory system), and a gate
// reads 1.8 MB of them: `low` is a projection of the pair (`gemv`, a segment a gate), and the two steps after it
// spread over many blocks (in one block the gate took 300 us).

// grid (outputs * rows, H / 64): the gated values in place, and each block's sum of their squares.
// low [rows, outputs, rank] BF16; forget, gain [G, H, rank] BF16; squares [outputs * rows, H / 64].
__global__ void __launch_bounds__(256) gate_apply_kernel(float* __restrict__ gated, const float* __restrict__ message,
                                                         const uint16_t* __restrict__ low, const uint16_t* __restrict__ forget,
                                                         const uint16_t* __restrict__ gain, float* __restrict__ squares,
                                                         int rows, int g0, int rank, int hidden) {
  __shared__ float fpart[256];
  __shared__ float gpart[256];
  __shared__ float scratch[8];
  const int po = blockIdx.x, t = threadIdx.x;
  const int j = po / rows, row = po % rows, g = g0 + j;
  const int h = 64 * blockIdx.y + t / 4, run = t % 4;
  const int width = rank / 4;
  const uint16_t* lw = low + (static_cast<int64_t>(row) * (gridDim.x / rows) + j) * rank;
  const uint16_t* fw = forget + (static_cast<int64_t>(g) * hidden + h) * rank;
  const uint16_t* gw = gain + (static_cast<int64_t>(g) * hidden + h) * rank;
  float f = 0.f, m = 0.f;
  for (int k = run * width; k < (run + 1) * width; ++k) {
    const float l = bf16_to_float(lw[k]);
    f = fmaf(bf16_to_float(fw[k]), l, f);
    m = fmaf(bf16_to_float(gw[k]), l, m);
  }
  fpart[t] = f;
  gpart[t] = m;
  __syncthreads();
  float square = 0.f;
  if (run == 0) {
    f = fpart[t] + fpart[t + 1] + fpart[t + 2] + fpart[t + 3];
    m = gpart[t] + gpart[t + 1] + gpart[t + 2] + gpart[t + 3];
    const int64_t at = static_cast<int64_t>(po) * hidden + h;
    const float value = (1.f - tanhf(f)) * gated[at] + tanhf(m) * message[at];
    gated[at] = value;
    square = value * value;
  }
  square = block_sum(square, scratch);
  if (t == 0) {
    squares[static_cast<int64_t>(po) * (hidden / 64) + blockIdx.y] = square;
  }
}

// grid (outputs * rows, H / 256), a thread a channel: out = BF16(rms(gated) * weight[output]).
__global__ void __launch_bounds__(256) gate_norm_kernel(const float* __restrict__ gated, const float* __restrict__ squares,
                                                        const float* __restrict__ weight, uint16_t* __restrict__ out, int rows, int hidden) {
  __shared__ float scratch[8];
  const int po = blockIdx.x, parts = hidden / 64;
  const int64_t at = static_cast<int64_t>(po) * hidden + blockIdx.y * 256 + threadIdx.x;
  const float value = gated[at], w = weight[static_cast<int64_t>(po / rows) * hidden + blockIdx.y * 256 + threadIdx.x];
  const float total = block_sum(static_cast<int>(threadIdx.x) < parts ? squares[static_cast<int64_t>(po) * parts + threadIdx.x] : 0.f, scratch);
  out[at] = float_to_bf16(value * rsqrtf(total / static_cast<float>(hidden) + kEps) * w);
}
// Rows come `steps` to a sequence (decode: one; verification: the pending token and its drafts), and a stream's
// convolution reads the sequence's last three inputs before the row's own: this forward's rows where there are any,
// the sequence's stored window (oldest first) before them. `Stream` is one channel of one row's view of that.
struct Stream {
  const uint16_t* x;       // the channel in the sequence's first row
  int64_t stride;          // elements between rows of x
  const uint16_t* window;  // the channel in the stored window's oldest row
  int64_t width;           // elements between the window's rows
  __device__ __forceinline__ uint16_t at(int step) const { return step >= 0 ? x[step * stride] : window[(3 + step) * width]; }
};

// taps . the inputs at steps j - 3 .. j (taps: oldest first), and the window after step j written to `after`: the
// stored window itself for a decode row (each value is read before its place is written), a snapshot for a verified one.
__device__ __forceinline__ float conv_step(const Stream& in, int j, uint16_t* __restrict__ after, const float* __restrict__ taps) {
  const uint16_t v3 = in.at(j - 3), v2 = in.at(j - 2), v1 = in.at(j - 1), v0 = in.at(j);
  after[0] = v2;
  after[in.width] = v1;
  after[2 * in.width] = v0;
  return taps[0] * bf16_to_float(v3) + taps[1] * bf16_to_float(v2) + taps[2] * bf16_to_float(v1) + taps[3] * bf16_to_float(v0);
}

// grid (rows, H / 256), a thread a channel (one block a row took 14 us: its taps and window are 50 KB of cold reads).
// c = BF16((conv(x) + x) / sqrt(5)) on the visit's stream at `offset`; then by mode
//   0: residual = c                                                                (a layer that opens a block)
//   1: residual = hidden + c
//   2: out = BF16(residual + c)                                                    (the MLP stream: the next hidden)
// state: [slots, visits, 3, width] BF16; a sequence's window is at slots[sequence]. windows: null (the window after
// the row replaces the stored one) or [sequences, win_steps, visits, 3, width], the window after each row.
__global__ void __launch_bounds__(256) conv_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ taps, uint16_t* __restrict__ state, const int32_t* __restrict__ slots,
    int64_t visit_elems, int64_t at, int64_t width, uint16_t* __restrict__ windows, int steps, int win_steps,
    const uint16_t* __restrict__ hidden_in, float* __restrict__ residual, uint16_t* __restrict__ out, int mode, int hidden) {
  const int row = blockIdx.x, h = blockIdx.y * 256 + threadIdx.x;
  const int seq = row / steps, j = row % steps;
  uint16_t* window = state + slots[seq] * visit_elems + at;
  uint16_t* after = windows == nullptr ? window : windows + (static_cast<int64_t>(seq) * win_steps + j) * visit_elems + at;
  const int64_t base = static_cast<int64_t>(row) * hidden;
  const Stream in{x + static_cast<int64_t>(seq) * steps * hidden + h, hidden, window + h, width};
  const float c = round_bf16((conv_step(in, j, after + h, taps + 4 * h) + bf16_to_float(x[base + h])) * rsqrtf(5.f));
  if (mode == 2) {
    out[base + h] = float_to_bf16(residual[base + h] + c);
  } else {
    residual[base + h] = mode == 0 ? c : bf16_to_float(hidden_in[base + h]) + c;
  }
}
__device__ __forceinline__ uint8_t float_to_e4m3(float f) {
  const float a = fabsf(f);
  const uint32_t sign = f < 0.f ? 0x80u : 0u;
  if (!(a < 448.f)) {
    return static_cast<uint8_t>(sign | 0x7Eu);
  }
  int e = ilogbf(fmaxf(a, 1e-30f));
  uint32_t code;
  if (e < -6) {
    code = static_cast<uint32_t>(rintf(ldexpf(a, 9)));  // subnormals: steps of 2^-9; 8 carries into the first normal
  } else {
    uint32_t m = static_cast<uint32_t>(rintf(ldexpf(a, 3 - e)));  // [8, 16]
    if (m == 16u) {
      m = 8u;
      ++e;
    }
    code = (static_cast<uint32_t>(e + 7) << 3) | (m - 8u);
  }
  return static_cast<uint8_t>(sign | min(code, 0x7Eu));
}

constexpr int kPrepThreads = 1024;

// grid (rows, 3): tasks 0 queries, relative states and sinks, 1 keys, 2 values.
//   pq [rows, >= 2 heads*128 + heads*d_rel + heads] BF16 = q | relative states | output gate | sink
//   pk, pv [rows, kv (+ kv_heads * 4 dynamic taps)] BF16
//   q = clip(pq_q * q_scale) (* tau);  rel = clip(pq_r * r_scale) (* tau) rounded to BF16 (FP32 [rows, heads, d_rel]:
//   the attention kernel multiplies it by the bias projection for the keys it reads);  sink FP32
//   k = clip(BF16((conv(pk) + pk) / n)),  v = rms(BF16((conv(pv) + pv) / n)), written to page `pages[row]` of the pools
//   taps: [kv, 4] the current token last (sliding), or [kv, 4] by backward distance plus the row's dynamic taps (full)
template <bool kFp8>
__global__ void __launch_bounds__(kPrepThreads) prep_kernel(
    const uint16_t* __restrict__ pq, int64_t pq_stride, const uint16_t* __restrict__ pk, const uint16_t* __restrict__ pv,
    int64_t pkv_stride, const float* __restrict__ q_scale, const float* __restrict__ r_scale,
    const float* __restrict__ k_taps, const float* __restrict__ v_taps, const int64_t* __restrict__ positions,
    uint16_t* __restrict__ state, const int32_t* __restrict__ slots, int64_t visit_elems, int64_t at, int64_t width,
    uint16_t* __restrict__ windows, int steps, int win_steps, uint8_t* __restrict__ k_pool, uint8_t* __restrict__ v_pool, const int32_t* __restrict__ pages,
    uint16_t* __restrict__ q_out, float* __restrict__ rel_out, float* __restrict__ sink_out,
    int heads, int kv_heads, int d_rel, bool full, float tau_floor, float tau_alpha, float qk_clip, float rel_clip) {
  extern __shared__ float shared[];
  float* vals = shared;                    // up to heads * 128
  float* rels = vals + heads * kHeadDim;   // heads * d_rel
  float* scale = rels + heads * d_rel;     // per head: the queries', then the relative states'
  const int row = blockIdx.x, task = blockIdx.y, t = threadIdx.x;
  const int q_size = heads * kHeadDim, r_size = heads * d_rel, kv = kv_heads * kHeadDim;
  if (task == 0) {
    const float tau = full ? 1.f + tau_alpha * logf(fmaxf(static_cast<float>(positions[row] + 1) / tau_floor, 1.f)) : 1.f;
    const uint16_t* q_row = pq + row * pq_stride;
    for (int i = t; i < q_size; i += kPrepThreads) {
      vals[i] = bf16_to_float(q_row[i]) * q_scale[i];
    }
    if (t < r_size) {
      rels[t] = bf16_to_float(q_row[q_size + t]) * r_scale[t];
    }
    if (t < heads) {
      sink_out[row * heads + t] = bf16_to_float(q_row[2 * q_size + r_size + t]);
    }
    __syncthreads();
    if (t < heads) {
      float ss = 0.f;
      for (int d = 0; d < kHeadDim; ++d) {
        ss = fmaf(vals[t * kHeadDim + d], vals[t * kHeadDim + d], ss);
      }
      scale[t] = fminf(qk_clip / (sqrtf(ss) + 1e-6f), 1.f) * tau;
      ss = 0.f;
      for (int d = 0; d < d_rel; ++d) {
        ss = fmaf(rels[t * d_rel + d], rels[t * d_rel + d], ss);
      }
      scale[heads + t] = fminf(rel_clip / (sqrtf(ss) + 1e-6f), 1.f) * tau;
    }
    __syncthreads();
    for (int i = t; i < q_size; i += kPrepThreads) {
      q_out[static_cast<int64_t>(row) * q_size + i] = float_to_bf16(vals[i] * scale[i / kHeadDim]);
    }
    if (t < r_size) {
      rel_out[static_cast<int64_t>(row) * r_size + t] = round_bf16(rels[t] * scale[heads + t / d_rel]);
    }
  } else {
    const bool key = task == 1;
    const uint16_t* x = (key ? pk : pv) + row * pkv_stride;
    const float* taps = key ? k_taps : v_taps;
    const int seq = row / steps, j = row % steps;
    const int64_t stream_at = at + (key ? 0 : kv);
    uint16_t* window = state + slots[seq] * visit_elems + stream_at;
    uint16_t* after = windows == nullptr ? window : windows + (static_cast<int64_t>(seq) * win_steps + j) * visit_elems + stream_at;
    const uint16_t* first = (key ? pk : pv) + static_cast<int64_t>(seq) * steps * pkv_stride;
    const float norm = rsqrtf(full ? 9.f : 5.f);
    for (int i = t; i < kv; i += kPrepThreads) {
      const uint16_t xv = x[i];
      const Stream in{first + i, pkv_stride, window + i, width};
      float conv;
      if (full) {  // taps by backward distance: lag 0 is the row's own value
        const int head = i / kHeadDim;
        float tap[4];
        for (int lag = 0; lag < 4; ++lag) {
          tap[3 - lag] = taps[4 * i + lag] + bf16_to_float(x[kv + 4 * head + lag]);
        }
        conv = conv_step(in, j, after + i, tap);
      } else {
        conv = conv_step(in, j, after + i, taps + 4 * i);
      }
      vals[i] = round_bf16((conv + bf16_to_float(xv)) * norm);
    }
    __syncthreads();
    if (t < kv_heads) {
      float ss = 0.f;
      for (int d = 0; d < kHeadDim; ++d) {
        ss = fmaf(vals[t * kHeadDim + d], vals[t * kHeadDim + d], ss);
      }
      scale[t] = key ? fminf(qk_clip / (sqrtf(ss) + 1e-6f), 1.f) : rsqrtf(ss / kHeadDim + kEps);
    }
    __syncthreads();
    uint8_t* pool = (key ? k_pool : v_pool) + static_cast<int64_t>(pages[row]) * kv * (kFp8 ? 1 : 2);
    for (int i = t; i < kv; i += kPrepThreads) {
      const float value = vals[i] * scale[i / kHeadDim];
      if constexpr (kFp8) {
        pool[i] = float_to_e4m3(round_bf16(value));
      } else {
        reinterpret_cast<uint16_t*>(pool)[i] = float_to_bf16(value);
      }
    }
  }
}
// grid (ceil(rows / R), experts / 4), 128 threads: warp w scores expert 4 blockIdx.y + w on the block's R rows,
// scores [rows, experts] = sigmoid(router x), each lane holding 16-byte runs of the router row, all loaded before the
// products. The last block of a row group to finish (`done`, reset by it) picks each row's `topk` largest scores +
// bias: ids int64 [rows, topk], weights = scale * score / their sum, FP32. router [E, H] BF16, H = 256 V. Cold router,
// one row 8.9 -> 7.3 us, 4 rows 10.8 -> 8.3, 16 rows 20.5 -> 11.3 against a scoring kernel (a thread summing a run
// of 36 values in one dependent chain) and a selection kernel.
template <int V, int R>
__global__ void __launch_bounds__(128) route_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ router,
                                                    const float* __restrict__ bias, float* __restrict__ scores, int* __restrict__ done,
                                                    int64_t* __restrict__ ids, float* __restrict__ weights, int rows, int experts,
                                                    int topk, float route_scale) {
  constexpr int kHidden = 256 * V;
  const int t = threadIdx.x, lane = t % 32, e = 4 * blockIdx.y + t / 32, first = blockIdx.x * R;
  const uint4* w = reinterpret_cast<const uint4*>(router + static_cast<int64_t>(e) * kHidden);
  uint4 wv[V];
#pragma unroll
  for (int i = 0; i < V; ++i) {
    wv[i] = __ldcs(w + lane + 32 * i);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = first + r;
    if (row < rows) {
      const uint4* xr = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(row) * kHidden);
      uint4 xv[V];
#pragma unroll
      for (int i = 0; i < V; ++i) {
        xv[i] = xr[lane + 32 * i];
      }
      float v = 0.f;
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const uint32_t a[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w}, b[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v = fmaf(__uint_as_float(a[j] << 16), __uint_as_float(b[j] << 16), v);
          v = fmaf(__uint_as_float(a[j] & 0xFFFF0000u), __uint_as_float(b[j] & 0xFFFF0000u), v);
        }
      }
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
      }
      if (lane == 0) {
        scores[static_cast<int64_t>(row) * experts + e] = 1.f / (1.f + expf(-v));
      }
    }
  }
  __shared__ int last;
  __threadfence();
  __syncthreads();
  if (t == 0) {
    last = atomicAdd(done + blockIdx.x, 1) == static_cast<int>(gridDim.y) - 1;
  }
  __syncthreads();
  if (!last) {
    return;
  }
  __threadfence();
  __shared__ __align__(16) float key[128];
  __shared__ float chosen[16];
  for (int r = 0; r < R && first + r < rows; ++r) {
    const int row = first + r;
    const float* score = scores + static_cast<int64_t>(row) * experts;
    const float mine = t < experts ? __ldcg(score + t) : 0.f;
    key[t] = t < experts ? mine + bias[t] : -INFINITY;  // absent experts rank last
    __syncthreads();
    if (t < experts) {
      const float k = key[t];
      int rank = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const float4 q = reinterpret_cast<const float4*>(key)[j];
        rank += (q.x > k || (q.x == k && 4 * j < t)) + (q.y > k || (q.y == k && 4 * j + 1 < t)) +
                (q.z > k || (q.z == k && 4 * j + 2 < t)) + (q.w > k || (q.w == k && 4 * j + 3 < t));
      }
      if (rank < topk) {
        ids[static_cast<int64_t>(row) * topk + rank] = t;
        chosen[rank] = mine;
      }
    }
    __syncthreads();
    if (t < topk) {
      float total = 0.f;
      for (int k = 0; k < topk; ++k) {
        total += chosen[k];
      }
      weights[static_cast<int64_t>(row) * topk + t] = chosen[t] / (total + 1e-20f) * route_scale;
    }
    __syncthreads();
  }
  if (t == 0) {
    done[blockIdx.x] = 0;
  }
}

// ------------------------------------------------------------------------------------------------- projections

constexpr int kGemvRows = 4;      // activation rows a launch takes
constexpr int kGemvChunks = 3;    // 512-byte pieces of its row a warp loads up front (16 bytes a lane each)
constexpr int kGemvPerBlock = 2;  // output rows a block
constexpr int kGemvMaxSplit = 16;  // warps a row at most: a row of at most 16 x 1536 bytes

// Programmatic dependent launch: a kernel launched with it may start while the one before it finishes, and reads what
// that one wrote only after `grid_wait`; `grid_launch` lets the next one start once all of this one's blocks run.
__device__ __forceinline__ void grid_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void grid_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// y[m, n] = BF16(scale_s sum_k x[m, k] w[n, k]), accumulated in FP32, for a matrix [n, k] of BF16 or of E4M3 with
// one FP32 scale s a stacked matrix (FP8: the activations stay BF16), and M rows: the attention projections,
// vocabulary head and message gates' low projection in decode (cuBLAS's one-row BF16 GEMV streams them at 165 GB/s).
// A row is split among S = ceil(row bytes / 1536) warps, each loading its 3 x 512 contiguous bytes (evict-first)
// before anything else, then reading the activations through L1 (no staging, no barrier before the products); the
// row's first warp adds the S partial sums in order. Against a warp a row looping over the whole row, BF16 at one row
// (decode graph, spark4): a visit's projections 168 -> 162 us, `wo` 61 -> 59, the head 3.86 -> 3.70 ms, the 192-row
// gate 9.9 -> 8.9, the 64-row gate 5.7 -> 3.9 (a plain streaming read of 40 MB: 165-168 us). E4M3 (spark5): a
// visit's query, key and value 67 + 2 x 13 us in three launches of `TrunkLinear`'s GEMV -> 82 in one, `wo` 32 -> 31,
// the head 1.87 -> 1.81 ms, and a sixteenth of the error (E4M3 activations: 2.6e-2 relative, BF16: 1.6e-3). The
// loads go out before `grid_wait`: launched with programmatic dependent launch the weights stream while the kernel
// before still runs.
// Each output row's sums do not depend on M (a row decodes alone as it verifies). The matrix may be up to three
// stacked ones (segments), each with its own M rows of x ([S, M, K]) or all reading the same ([M, K]): output rows
// from `first` are the second segment, from `second` the third (a visit's query, key and value projections, or its
// message gates, in one launch).
template <int M, bool FP8>
__global__ void __launch_bounds__(kGemvMaxSplit * kGemvPerBlock * 32) gemv_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
                                                                                  const float* __restrict__ scales, uint16_t* __restrict__ out,
                                                                                  int n, int k, int first, int second, int64_t set_stride) {
  constexpr int kVec = FP8 ? 16 : 8;  // weights a 16-byte load holds
  grid_launch();
  const int split = blockDim.x / (32 * kGemvPerBlock);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, part = warp % split;
  const int row = blockIdx.x * kGemvPerBlock + warp / split;
  const int c0 = part * kGemvChunks * 32 * kVec + lane * kVec;
  const uint8_t* wr = w + (static_cast<int64_t>(row < n ? row : n - 1) * k + c0) * (FP8 ? 1 : 2);
  float4 wv[kGemvChunks];
#pragma unroll
  for (int i = 0; i < kGemvChunks; ++i) {
    wv[i] = c0 + i * 32 * kVec < k ? __ldcs(reinterpret_cast<const float4*>(wr + i * 512)) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  grid_wait();
  const int segment = (row >= first) + (row >= second);
  const uint16_t* xs = x + segment * set_stride + c0;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < kGemvChunks; ++i) {
    if (c0 + i * 32 * kVec < k) {
      float wf[kVec];
      if constexpr (FP8) {
        const __nv_fp8x2_storage_t* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&wv[i]);
#pragma unroll
        for (int j = 0; j < kVec / 2; ++j) {
          const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pairs[j], __NV_E4M3)));
          wf[2 * j] = f.x;
          wf[2 * j + 1] = f.y;
        }
      } else {
        const uint16_t* wh = reinterpret_cast<const uint16_t*>(&wv[i]);
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          wf[j] = bf16_to_float(wh[j]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int h = 0; h < kVec / 8; ++h) {
          const uint4 xv = __ldg(reinterpret_cast<const uint4*>(xs + m * k + i * 32 * kVec + h * 8));
          const uint16_t* xh = reinterpret_cast<const uint16_t*>(&xv);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[m] = fmaf(wf[h * 8 + j], bf16_to_float(xh[j]), acc[m]);
          }
        }
      }
    }
  }
  __shared__ float partial[kGemvMaxSplit * kGemvPerBlock][M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      partial[warp][m] = v;
    }
  }
  __syncthreads();
  if (part == 0 && lane < M && row < n) {
    float v = 0.f;
    for (int p = 0; p < split; ++p) {
      v += partial[warp + p][lane];
    }
    out[static_cast<int64_t>(lane) * n + row] = float_to_bf16(FP8 ? v * scales[segment] : v);
  }
}

// -------------------------------------------------------------------------------------------------- experts
//
// The experts' products and their activation run on the tensor cores (`kernels/inkling_prefill.py`); the tokens'
// weighted sums after them are one pass over the (token, expert) pairs instead of PyTorch's dozen.

// grid (tokens, ceil(H / 256)), a thread a channel: out[t] = BF16(sum_k weights[t, k] down[where[t, k]]), the token's
// experts' outputs from their rows (where null: rows t topk .. t topk + topk - 1).
__global__ void __launch_bounds__(256) combine_kernel(const uint16_t* __restrict__ down, const int64_t* __restrict__ where,
                                                      const float* __restrict__ weights, uint16_t* __restrict__ out, int topk, int hidden) {
  const int64_t t = blockIdx.x;
  const int h = blockIdx.y * 256 + threadIdx.x;
  if (h >= hidden) {
    return;
  }
  float v = 0.f;
  for (int k = 0; k < topk; ++k) {
    const int64_t row = where != nullptr ? where[t * topk + k] : t * topk + k;
    v = fmaf(weights[t * topk + k], bf16_to_float(down[row * hidden + h]), v);
  }
  out[t * hidden + h] = float_to_bf16(v);
}

void check(bool ok, const char* what) { TORCH_CHECK(ok, "inkling kernels: ", what); }

}  // namespace

// q [R, 24, 128] BF16; rel [R, 24, d_rel] FP32 and proj [extent, d_rel] BF16, the relative-position states and the
// projection to their bias by backward distance (both empty: no bias); sink [R, 24] FP32; k_pool, v_pool
// [pages, 1, 8, 128] float8 or BF16; indptr [S + 1], indices int32 (each sequence's pages by position); row_seq,
// row_len [R] int32 (a row's sequence and how many of its keys the row sees, its own last); window 0 = every key; gate
// (may be empty): BF16 rows whose values from `gate_offset` gate the output by their sigmoid; rows come `steps` to a
// sequence (consecutive rows of one sequence: they share their key loads). -> [R, 24, 128] BF16.
torch::Tensor attention(torch::Tensor q, torch::Tensor rel, torch::Tensor proj, torch::Tensor sink, torch::Tensor k_pool, torch::Tensor v_pool,
                        torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_seq, torch::Tensor row_len,
                        int64_t window, int64_t splits, double scale, torch::Tensor gate, int64_t gate_offset, int64_t steps) {
  const int rows = q.size(0), heads = q.size(1), kv_heads = k_pool.size(-2);
  const bool fp8 = k_pool.scalar_type() != torch::kBFloat16;
  check(q.is_cuda() && q.is_contiguous() && q.scalar_type() == torch::kBFloat16 && q.size(2) == kHeadDim, "q must be contiguous BF16 [R, heads, 128]");
  check(heads == kv_heads * kGroup && k_pool.size(-1) == kHeadDim && k_pool.element_size() == (fp8 ? 1 : 2) &&
            v_pool.scalar_type() == k_pool.scalar_type() && v_pool.sizes() == k_pool.sizes() && k_pool.is_contiguous() &&
            v_pool.is_contiguous(), "unexpected K/V geometry");
  const bool biased = proj.numel() > 0;
  const int d_rel = biased ? proj.size(1) : 0, extent = biased ? proj.size(0) : 0;
  check(!biased || (rel.is_contiguous() && rel.scalar_type() == torch::kFloat32 && rel.numel() == static_cast<int64_t>(rows) * heads * d_rel &&
                    proj.is_contiguous() && proj.scalar_type() == torch::kBFloat16 && d_rel == kRelMax),
        "rel must be FP32 [R, heads, 16] and proj BF16 [extent, 16]");
  check(sink.is_contiguous() && sink.scalar_type() == torch::kFloat32 && sink.numel() == static_cast<int64_t>(rows) * heads, "sink must be FP32 [R, heads]");
  check(indptr.scalar_type() == torch::kInt32 && indices.scalar_type() == torch::kInt32 && row_seq.scalar_type() == torch::kInt32 &&
            row_len.scalar_type() == torch::kInt32 && row_seq.numel() == rows && row_len.numel() == rows && indptr.is_contiguous() &&
            indices.is_contiguous() && row_seq.is_contiguous() && row_len.is_contiguous(), "index tensors must be contiguous int32");
  check(splits >= 1 && splits <= kMaxSplits, "splits must be 1 .. 128");
  check(steps >= 1 && rows % steps == 0, "rows must come steps to a sequence");
  const bool gated = gate.numel() > 0;
  check(!gated || (gate.scalar_type() == torch::kBFloat16 && gate.stride(1) == 1 && gate.size(0) == rows &&
                   gate.size(1) >= gate_offset + heads * kHeadDim), "gate must be BF16 rows [R, >= offset + heads * 128]");
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto partial = torch::empty({static_cast<int64_t>(rows) * heads * splits, kPartial}, q.options().dtype(torch::kFloat32));
  auto out = torch::empty_like(q);
  // a block's rows: the largest divisor of `steps` the kernel holds (BF16 K/V stage twice the bytes, so eight rows'
  // queries and weights would pass the 48 KB of static shared memory)
  const int most = fp8 ? 8 : 4;
  int group = 1;
  for (int d = 1; d <= most; ++d) {
    if (steps % d == 0) {
      group = d;
    }
  }
  const dim3 grid(rows / group * kv_heads, splits);
  const auto* qp = reinterpret_cast<const uint16_t*>(q.data_ptr());
  const float* relp = biased ? rel.data_ptr<float>() : nullptr;
  const auto* projp = biased ? reinterpret_cast<const uint16_t*>(proj.data_ptr()) : nullptr;
  const auto* kp = reinterpret_cast<const uint8_t*>(k_pool.data_ptr());
  const auto* vp = reinterpret_cast<const uint8_t*>(v_pool.data_ptr());
#define INKLING_ATTENTION(FP8, R)                                                                                              \
  attention_kernel<FP8, R><<<grid, kAttnThreads, 0, stream>>>(qp, relp, projp, kp, vp, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(), \
                                                              row_seq.data_ptr<int32_t>(), row_len.data_ptr<int32_t>(),           \
                                                              partial.data_ptr<float>(), kv_heads, extent, window, splits, group, scale)
  if (fp8) {
    if (group == 1) {
      INKLING_ATTENTION(true, 1);
    } else if (group == 2) {
      INKLING_ATTENTION(true, 2);
    } else if (group <= 4) {
      INKLING_ATTENTION(true, 4);
    } else {
      INKLING_ATTENTION(true, 8);
    }
  } else {
    if (group == 1) {
      INKLING_ATTENTION(false, 1);
    } else if (group == 2) {
      INKLING_ATTENTION(false, 2);
    } else {
      INKLING_ATTENTION(false, 4);
    }
  }
#undef INKLING_ATTENTION
  attention_merge_kernel<<<rows * heads, kHeadDim, 0, stream>>>(
      partial.data_ptr<float>(), sink.data_ptr<float>(),
      gated ? reinterpret_cast<const uint16_t*>(gate.data_ptr()) + gate_offset : nullptr, gated ? gate.stride(0) : 0,
      reinterpret_cast<uint16_t*>(out.data_ptr()), splits, heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The decode glue's entry points (the kernels' comments have the arithmetic). An empty tensor stands for "none".
torch::Tensor gemv(torch::Tensor x, torch::Tensor w, int64_t first, int64_t second, torch::Tensor scales);

torch::Tensor mix(torch::Tensor states, int64_t count, torch::Tensor partial, torch::Tensor queries, int64_t outputs,
                  torch::Tensor down, torch::Tensor forget, torch::Tensor gain, torch::Tensor sources,
                  torch::Tensor msg_queries, int64_t g0, torch::Tensor weight) {
  const int rows = partial.size(0), hidden = partial.size(1);
  const bool bf16 = partial.scalar_type() == torch::kBFloat16, gated = down.numel() > 0;
  check(partial.is_cuda() && partial.is_contiguous() && (bf16 || partial.scalar_type() == torch::kFloat32), "partial must be contiguous BF16 or FP32");
  check(hidden == kMixThreads * kMixPer, "unexpected hidden size");
  check(states.scalar_type() == torch::kFloat32 && states.is_contiguous() && states.size(0) == rows && states.size(2) == hidden &&
            count >= 0 && count <= states.size(1) && count <= 5, "states must be FP32 [R, slots >= count, H], count <= 5");
  check(queries.scalar_type() == torch::kFloat32 && queries.is_contiguous() && queries.size(1) == hidden, "queries must be FP32 [Q, H]");
  check(weight.scalar_type() == torch::kFloat32 && weight.is_contiguous() && weight.size(0) >= outputs && weight.size(1) == hidden, "weight must be FP32 [outputs, H]");
  int rank = 64;
  if (gated) {
    rank = down.size(1);
    check(down.scalar_type() == torch::kBFloat16 && down.is_contiguous() && down.size(2) == 2 * hidden && rank == 64 &&
              hidden % 64 == 0 && g0 >= 0 && g0 + outputs <= down.size(0) && outputs <= 3, "unexpected gate shapes");
    check(forget.scalar_type() == torch::kBFloat16 && gain.scalar_type() == torch::kBFloat16 && forget.is_contiguous() && gain.is_contiguous() &&
              sources.scalar_type() == torch::kFloat32 && sources.is_contiguous() && sources.size(0) == rows && sources.size(1) == 2 &&
              msg_queries.scalar_type() == torch::kFloat32 && msg_queries.is_contiguous(), "unexpected gate tensors");
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto out = torch::empty({outputs, rows, hidden}, partial.options().dtype(torch::kBFloat16));
  const auto f32 = partial.options().dtype(torch::kFloat32);
  torch::Tensor value, message, pair, squares;
  if (gated) {
    value = torch::empty({outputs, rows, hidden}, f32);
    message = torch::empty({outputs, rows, hidden}, f32);
    pair = torch::empty({outputs, rows, 2 * hidden}, partial.options().dtype(torch::kBFloat16));
    squares = torch::empty({outputs, rows, hidden / 64}, f32);
  }
  const dim3 grid(rows, outputs);
  const float* sp = gated ? sources.data_ptr<float>() : nullptr;
  const float* mp = gated ? msg_queries.data_ptr<float>() : nullptr;
  float* vp = gated ? value.data_ptr<float>() : nullptr;
  float* gp = gated ? message.data_ptr<float>() : nullptr;
  auto* pp = gated ? reinterpret_cast<uint16_t*>(pair.data_ptr()) : nullptr;
  const float* wp = weight.data_ptr<float>();
  auto* op = reinterpret_cast<uint16_t*>(out.data_ptr());
#define HIBANA_MIX(BF16, COUNT)                                                                                              \
  mix_kernel<BF16, COUNT><<<grid, kMixThreads, 0, stream>>>(states.data_ptr<float>(), states.size(1), partial.data_ptr(),  \
                                                            queries.data_ptr<float>(), queries.size(0), sp, mp, g0, vp, gp, pp, wp, op)
#define HIBANA_MIX_COUNT(COUNT)                                                                                              \
  case COUNT:                                                                                                                \
    if (bf16) {                                                                                                              \
      HIBANA_MIX(true, COUNT);                                                                                               \
    } else {                                                                                                                 \
      HIBANA_MIX(false, COUNT);                                                                                              \
    }                                                                                                                        \
    break
  switch (count) {  // an instance a count: the states stay in registers
    HIBANA_MIX_COUNT(0);
    HIBANA_MIX_COUNT(1);
    HIBANA_MIX_COUNT(2);
    HIBANA_MIX_COUNT(3);
    HIBANA_MIX_COUNT(4);
    default:
      HIBANA_MIX_COUNT(5);
  }
#undef HIBANA_MIX_COUNT
#undef HIBANA_MIX
  if (gated) {
    const int pairs = rows * outputs;
    // low [rows, outputs, rank]: every gate's pair through its own down projection, in one launch for a few rows
    auto downs = down.narrow(0, g0, outputs);
    auto low = rows <= kGemvRows ? gemv(pair, downs.reshape({outputs * rank, 2 * hidden}), outputs > 1 ? rank : outputs * rank,
                                        outputs > 2 ? 2 * rank : outputs * rank, pair.new_empty(0))
                                 : torch::bmm(pair, downs.transpose(1, 2)).transpose(0, 1).contiguous();
    gate_apply_kernel<<<dim3(pairs, hidden / 64), 256, 0, stream>>>(vp, gp, reinterpret_cast<const uint16_t*>(low.data_ptr()),
                                                                   reinterpret_cast<const uint16_t*>(forget.data_ptr()),
                                                                   reinterpret_cast<const uint16_t*>(gain.data_ptr()),
                                                                   squares.data_ptr<float>(), rows, g0, rank, hidden);
    gate_norm_kernel<<<dim3(pairs, hidden / 256), 256, 0, stream>>>(vp, squares.data_ptr<float>(), wp, op, rows, hidden);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The index checks of a call whose rows come `steps` to a sequence: slots [R / steps] int32, and windows, empty (the
// stored windows advance) or BF16 [>= R / steps, >= steps, visits, 3, width] (the window after each row).
void check_steps(int rows, int64_t steps, const torch::Tensor& slots, const torch::Tensor& windows, const torch::Tensor& state) {
  check(steps >= 1 && rows % steps == 0 && slots.scalar_type() == torch::kInt32 && slots.numel() == rows / steps, "slots must be int32, one a sequence");
  check(windows.numel() == 0 ? steps == 1
                             : (windows.scalar_type() == torch::kBFloat16 && windows.is_contiguous() && windows.dim() == 5 &&
                                windows.size(0) >= rows / steps && windows.size(1) >= steps && windows.size(2) == state.size(1) &&
                                windows.size(3) == 3 && windows.size(4) == state.size(3)),
        "rows of a sequence need windows [sequences, steps, visits, 3, width]");
}

// x [R, H] BF16; taps [H, 4] FP32; state [slots, visits, 3, width] BF16; slots, windows, steps: `check_steps`. mode
// 0: -> residual [R, H] FP32; mode 1: `hidden` [R, H] BF16 -> residual; mode 2: `residual` [R, H] FP32 -> [R, H] BF16.
torch::Tensor conv(torch::Tensor x, torch::Tensor taps, torch::Tensor state, torch::Tensor slots, int64_t visit, int64_t offset,
                   torch::Tensor hidden_in, torch::Tensor residual, int64_t mode, torch::Tensor windows, int64_t steps) {
  const int rows = x.size(0), hidden = x.size(1);
  check(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && hidden % 256 == 0, "x must be contiguous BF16 [R, 256 n]");
  check(taps.scalar_type() == torch::kFloat32 && taps.is_contiguous() && taps.size(0) == hidden && taps.size(1) == 4, "taps must be FP32 [H, 4]");
  check(state.scalar_type() == torch::kBFloat16 && state.is_contiguous() && state.dim() == 4 && state.size(2) == 3 &&
            visit >= 0 && visit < state.size(1) && offset + hidden <= state.size(3), "state must be BF16 [slots, visits, 3, width]");
  check_steps(rows, steps, slots, windows, state);
  check(mode == 2 ? (residual.scalar_type() == torch::kFloat32 && residual.is_contiguous() && residual.numel() == x.numel())
                  : mode == 0 || (hidden_in.scalar_type() == torch::kBFloat16 && hidden_in.is_contiguous() && hidden_in.numel() == x.numel()),
        "mode 2 reads an FP32 residual, mode 1 the BF16 hidden");
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int64_t width = state.size(3), visit_elems = state.size(1) * 3 * width;
  torch::Tensor out;
  if (mode == 2) {
    out = torch::empty_like(x);
  } else {
    residual = torch::empty({rows, hidden}, x.options().dtype(torch::kFloat32));
  }
  conv_kernel<<<dim3(rows, hidden / 256), 256, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x.data_ptr()), taps.data_ptr<float>(), reinterpret_cast<uint16_t*>(state.data_ptr()),
      slots.data_ptr<int32_t>(), visit_elems, visit * 3 * width + offset, width,
      windows.numel() ? reinterpret_cast<uint16_t*>(windows.data_ptr()) : nullptr, steps, windows.numel() ? windows.size(1) : 1,
      mode == 1 ? reinterpret_cast<const uint16_t*>(hidden_in.data_ptr()) : nullptr, residual.data_ptr<float>(),
      mode == 2 ? reinterpret_cast<uint16_t*>(out.data_ptr()) : nullptr, mode, hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mode == 2 ? out : residual;
}

// -> (q [R, heads, 128] BF16, rel [R, heads, d_rel] FP32, sink [R, heads] FP32); K and V go to `pages` of the pools.
// pq, pk, pv: BF16 rows (they may be column ranges of one projection). slots, windows, steps: `check_steps`.
std::vector<torch::Tensor> prep(torch::Tensor pq, torch::Tensor pk, torch::Tensor pv, torch::Tensor q_scale, torch::Tensor r_scale,
                                torch::Tensor k_taps, torch::Tensor v_taps, torch::Tensor positions,
                                torch::Tensor state, torch::Tensor slots, int64_t visit, torch::Tensor k_pool, torch::Tensor v_pool,
                                torch::Tensor pages, bool full, double tau_floor, double tau_alpha, double qk_clip, double rel_clip,
                                torch::Tensor windows, int64_t steps) {
  const int rows = pq.size(0), heads = q_scale.size(0), d_rel = r_scale.size(1), kv_heads = k_pool.size(-2);
  const int kv = kv_heads * kHeadDim;
  const bool fp8 = k_pool.scalar_type() != torch::kBFloat16;
  check(pq.is_cuda() && pq.stride(1) == 1 && pk.stride(1) == 1 && pv.stride(1) == 1 && pk.stride(0) == pv.stride(0) &&
            pq.scalar_type() == torch::kBFloat16 && pk.scalar_type() == torch::kBFloat16 && pv.scalar_type() == torch::kBFloat16,
        "projections must be BF16 rows");
  check(pq.size(1) >= 2 * heads * kHeadDim + heads * d_rel + heads && pk.size(1) >= kv + (full ? 4 * kv_heads : 0) && pv.size(1) == pk.size(1) &&
            pk.size(0) == rows && pv.size(0) == rows, "unexpected projection widths");
  check(q_scale.scalar_type() == torch::kFloat32 && q_scale.is_contiguous() && q_scale.size(1) == kHeadDim && r_scale.scalar_type() == torch::kFloat32 &&
            r_scale.is_contiguous() && heads * d_rel <= kPrepThreads, "unexpected scales");
  check(k_taps.scalar_type() == torch::kFloat32 && k_taps.is_contiguous() && k_taps.numel() == 4 * kv && v_taps.scalar_type() == torch::kFloat32 &&
            v_taps.is_contiguous() && v_taps.numel() == 4 * kv, "taps must be FP32, 4 a channel");
  check(state.scalar_type() == torch::kBFloat16 && state.is_contiguous() && state.dim() == 4 && state.size(2) == 3 && state.size(3) >= 2 * kv &&
            visit >= 0 && visit < state.size(1), "state must be BF16 [slots, visits, 3, width]");
  check_steps(rows, steps, slots, windows, state);
  check(pages.scalar_type() == torch::kInt32 && pages.numel() == rows && positions.scalar_type() == torch::kInt64 && positions.numel() == rows,
        "pages int32 and positions int64, [R]");
  check(k_pool.size(-1) == kHeadDim && k_pool.element_size() == (fp8 ? 1 : 2) && v_pool.scalar_type() == k_pool.scalar_type() &&
            heads * kHeadDim + heads <= 4096, "unexpected K/V geometry");
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto q = torch::empty({rows, heads, kHeadDim}, pq.options());
  auto rel = torch::empty({rows, heads, d_rel}, pq.options().dtype(torch::kFloat32));
  auto sink = torch::empty({rows, heads}, pq.options().dtype(torch::kFloat32));
  const int64_t width = state.size(3), visit_elems = state.size(1) * 3 * width;
  const size_t shared = (heads * kHeadDim + heads * d_rel + 2 * heads) * sizeof(float);
  const dim3 grid(rows, 3);
#define HIBANA_PREP(FP8)                                                                                                     \
  prep_kernel<FP8><<<grid, kPrepThreads, shared, stream>>>(                                                                  \
      reinterpret_cast<const uint16_t*>(pq.data_ptr()), pq.stride(0), reinterpret_cast<const uint16_t*>(pk.data_ptr()),      \
      reinterpret_cast<const uint16_t*>(pv.data_ptr()), pk.stride(0), q_scale.data_ptr<float>(), r_scale.data_ptr<float>(),  \
      k_taps.data_ptr<float>(), v_taps.data_ptr<float>(),                                                                    \
      positions.data_ptr<int64_t>(), reinterpret_cast<uint16_t*>(state.data_ptr()), slots.data_ptr<int32_t>(), visit_elems,  \
      visit * 3 * width, width, windows.numel() ? reinterpret_cast<uint16_t*>(windows.data_ptr()) : nullptr, steps,          \
      windows.numel() ? windows.size(1) : 1, reinterpret_cast<uint8_t*>(k_pool.data_ptr()),                                  \
      reinterpret_cast<uint8_t*>(v_pool.data_ptr()),                                                                         \
      pages.data_ptr<int32_t>(), reinterpret_cast<uint16_t*>(q.data_ptr()), rel.data_ptr<float>(),                           \
      sink.data_ptr<float>(), heads, kv_heads, d_rel, full, tau_floor, tau_alpha, qk_clip, rel_clip)
  if (fp8) {
    HIBANA_PREP(true);
  } else {
    HIBANA_PREP(false);
  }
#undef HIBANA_PREP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, rel, sink};
}

// x [R, H] BF16; router [E, H] BF16, bias [E] FP32; done int32 [>= ceil(R / 4)], zero between calls (the kernel resets
// it) -> (ids int64 [R, topk], weights FP32 [R, topk]).
std::vector<torch::Tensor> route(torch::Tensor x, torch::Tensor router, torch::Tensor bias, torch::Tensor done, int64_t topk, double scale) {
  constexpr int kRows = 4;  // rows a block scores against the router rows it holds
  const int rows = x.size(0), hidden = x.size(1), experts = router.size(0);
  check(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x must be contiguous BF16");
  check(router.scalar_type() == torch::kBFloat16 && router.is_contiguous() && router.size(1) == hidden && experts <= 128 && experts % 4 == 0 &&
            hidden == 2304 && bias.scalar_type() == torch::kFloat32 && bias.numel() == experts && topk <= 16 && topk <= experts,
        "unexpected router shapes");
  check(done.scalar_type() == torch::kInt32 && done.numel() >= (rows <= 16 ? rows : (rows + kRows - 1) / kRows), "done must be int32 [a counter a block group]");
  check(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(router.data_ptr()) % 16 == 0, "x and router must be 16-byte aligned");
  auto ids = torch::empty({rows, topk}, x.options().dtype(torch::kInt64));
  auto weights = torch::empty({rows, topk}, x.options().dtype(torch::kFloat32));
  auto scores = torch::empty({rows, experts}, x.options().dtype(torch::kFloat32));
  if (rows > 0) {
    const dim3 grid((rows + kRows - 1) / kRows, experts / 4);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
    const auto* rp = reinterpret_cast<const uint16_t*>(router.data_ptr());
    if (rows <= 16) {  // a block a row: the last block of a group selects its rows one after another
      route_kernel<9, 1><<<dim3(rows, experts / 4), 128, 0, stream>>>(xp, rp, bias.data_ptr<float>(), scores.data_ptr<float>(), done.data_ptr<int>(),
                                                   ids.data_ptr<int64_t>(), weights.data_ptr<float>(), rows, experts, topk, scale);
    } else {
      route_kernel<9, kRows><<<grid, 128, 0, stream>>>(xp, rp, bias.data_ptr<float>(), scores.data_ptr<float>(), done.data_ptr<int>(),
                                                       ids.data_ptr<int64_t>(), weights.data_ptr<float>(), rows, experts, topk, scale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {ids, weights};
}

// x [M, K] or [S, M, K] BF16 and w [N, K] BF16, or E4M3 with `scales` (FP32, one a segment), M <= 4 -> [M, N] BF16:
// x w^T, w's rows from `first` and from `second` (N for none) the second and third segments, which read x's second
// and third sets when x has them. Launched with programmatic dependent launch (see `grid_wait`).
torch::Tensor gemv(torch::Tensor x, torch::Tensor w, int64_t first, int64_t second, torch::Tensor scales) {
  check(x.is_cuda() && (x.dim() == 2 || x.dim() == 3) && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x must be contiguous BF16 [(S,) M, K]");
  const int sets = x.dim() == 3 ? x.size(0) : 1, m = x.size(-2), k = x.size(-1), n = w.size(0);
  const bool fp8 = w.scalar_type() == torch::kFloat8_e4m3fn;
  check(w.dim() == 2 && w.is_contiguous() && (fp8 || w.scalar_type() == torch::kBFloat16) && w.size(1) == k, "w must be contiguous BF16 or E4M3 [N, K]");
  const int segments = 1 + (first < n) + (second < n);
  check(!fp8 || (scales.scalar_type() == torch::kFloat32 && scales.is_contiguous() && scales.numel() == segments), "an E4M3 matrix needs an FP32 scale a segment");
  const int row_bytes = k * (fp8 ? 1 : 2);
  check(m >= 1 && m <= kGemvRows && k % (fp8 ? 16 : 8) == 0 && row_bytes <= kGemvMaxSplit * kGemvChunks * 512, "unexpected projection shape");
  check(first <= second && (sets == 1 || sets == segments), "x's sets do not match the segments");
  auto out = torch::empty({m, n}, x.options());
  const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const uint8_t*>(w.data_ptr());
  const float* sp = fp8 ? scales.data_ptr<float>() : nullptr;
  auto* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  const int split = (row_bytes + kGemvChunks * 512 - 1) / (kGemvChunks * 512);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((n + kGemvPerBlock - 1) / kGemvPerBlock);
  cfg.blockDim = dim3(split * kGemvPerBlock * 32);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const int ni = n, ki = k, fi = first, si = second;
  const int64_t stride = sets > 1 ? static_cast<int64_t>(m) * k : 0;
#define HIBANA_GEMV(M_)                                                                                         \
  C10_CUDA_CHECK(fp8 ? cudaLaunchKernelEx(&cfg, gemv_kernel<M_, true>, xp, wp, sp, op, ni, ki, fi, si, stride)  \
                     : cudaLaunchKernelEx(&cfg, gemv_kernel<M_, false>, xp, wp, sp, op, ni, ki, fi, si, stride))
  switch (m) {
    case 1: HIBANA_GEMV(1); break;
    case 2: HIBANA_GEMV(2); break;
    case 3: HIBANA_GEMV(3); break;
    default: HIBANA_GEMV(4); break;
  }
#undef HIBANA_GEMV
  return out;
}

// x [R, W] BF16 or FP32, weight [W] FP32 or empty -> x rsqrt(mean(x^2) + eps) (* weight), BF16 or FP32 [R, W];
// with `into` (FP32 [R, slots, W]) written to into[:, slot] instead.
torch::Tensor rms(torch::Tensor x, torch::Tensor weight, double eps, bool out_bf16, torch::Tensor into, int64_t slot) {
  const bool bf16 = x.scalar_type() == torch::kBFloat16, weighted = weight.numel() > 0, placed = into.numel() > 0;
  check(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && (bf16 || x.scalar_type() == torch::kFloat32) && x.size(1) <= kMixThreads * kMixPer,
        "x must be contiguous BF16 or FP32 [R, W <= 2304]");
  const int64_t rows = x.size(0), width = x.size(1);
  check(!weighted || (weight.scalar_type() == torch::kFloat32 && weight.is_contiguous() && weight.numel() == width), "weight must be FP32 [W]");
  check(!placed || (!out_bf16 && into.scalar_type() == torch::kFloat32 && into.is_contiguous() && into.dim() == 3 && into.size(0) == rows &&
                    into.size(2) == width && slot >= 0 && slot < into.size(1)), "into must be FP32 [R, slots, W]");
  auto out = placed ? into : torch::empty({rows, width}, x.options().dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32));
  if (rows == 0) {
    return out;
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  const float* wp = weighted ? weight.data_ptr<float>() : nullptr;
  void* op = placed ? static_cast<void*>(into.data_ptr<float>() + slot * width) : out.data_ptr();
  const int64_t stride = placed ? into.size(1) * width : width;
  const unsigned grid = static_cast<unsigned>(rows);
  if (bf16 && out_bf16) {
    rms_kernel<true, true><<<grid, kMixThreads, 0, stream>>>(x.data_ptr(), wp, op, stride, width, eps);
  } else if (bf16) {
    rms_kernel<true, false><<<grid, kMixThreads, 0, stream>>>(x.data_ptr(), wp, op, stride, width, eps);
  } else if (out_bf16) {
    rms_kernel<false, true><<<grid, kMixThreads, 0, stream>>>(x.data_ptr(), wp, op, stride, width, eps);
  } else {
    rms_kernel<false, false><<<grid, kMixThreads, 0, stream>>>(x.data_ptr(), wp, op, stride, width, eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// down [P, H] BF16; weights [T, k] FP32; where [T, k] int64 rows of down, or empty for a token's rows in order
// -> [T, H] BF16.
torch::Tensor combine(torch::Tensor down, torch::Tensor where, torch::Tensor weights) {
  check(down.is_cuda() && down.dim() == 2 && down.is_contiguous() && down.scalar_type() == torch::kBFloat16, "down must be contiguous BF16 [P, H]");
  check(weights.dim() == 2 && weights.is_contiguous() && weights.scalar_type() == torch::kFloat32 && weights.numel() == down.size(0),
        "weights must be FP32 [T, k] over down's rows");
  const bool placed = where.numel() > 0;
  check(!placed || (where.is_contiguous() && where.scalar_type() == torch::kInt64 && where.sizes() == weights.sizes()), "where must be int64 [T, k]");
  auto out = torch::empty({weights.size(0), down.size(1)}, down.options());
  if (weights.size(0) > 0) {
    const dim3 grid(static_cast<unsigned>(weights.size(0)), static_cast<unsigned>((down.size(1) + 255) / 256));
    combine_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint16_t*>(down.data_ptr()), placed ? where.data_ptr<int64_t>() : nullptr, weights.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(weights.size(1)), static_cast<int>(down.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &attention);
  m.def("mix", &mix);
  m.def("conv", &conv);
  m.def("prep", &prep);
  m.def("route", &route);
  m.def("gemv", &gemv);
  m.def("rms", &rms);
  m.def("combine", &combine);
}
