"""The DSpark adapter for Inkling's bundled block drafter (`model/inkling/dspark.py`).

The draft's context is a per-page row of the target's KV pool, written here from the target's hidden states as
positions commit (prefill chunks, decode steps, accepted rounds), so the adapter keeps no ring and no feature
window: a request resumes from a cached prefix with its draft context in the pages it matched. What it keeps per
request is one row, the recurrent head's read of the last committed position. Drafts are the draft's argmax: point
masses, which a sampled request verifies by rejection sampling like any other.
"""

from __future__ import annotations

import torch

from hibana.engine import graph_set
from hibana.engine.contracts import Capabilities, FeatureRequest, ProposerRequirements


class DSparkProposer:
    def __init__(self, draft, context, k: int, slots: int):
        """`draft`: the model's `InklingDraft`, its pages registered with the KV pool; `slots`: the state pool's."""
        if k != draft.block - 1:
            raise ValueError(f"DSpark drafts its whole block: {draft.block - 1} tokens (verify fewer with HIBANA_SPEC_DEPTHS)")
        if draft.cos.shape[0] < context.max_context + draft.block:
            raise ValueError(f"the DSpark drafter covers {draft.cos.shape[0] - draft.block} positions, the engine serves {context.max_context}")
        self.draft, self.context, self.k = draft, context, k
        # the recurrent head's context of each request, by its state slot
        self.head_context = torch.zeros((slots, draft.hidden), dtype=torch.bfloat16, device=context.device)
        self.sizes: dict[int, tuple[int, ...]] = {}  # verify depth -> the batch buckets with a draft graph
        self.static, self.graphs = {}, {}
        self.pool = torch.cuda.graph_pool_handle() if context.device.type == "cuda" else None
        self.requirements = ProposerRequirements(
            "dspark", FeatureRequest(final_norm=True, layers=draft.features), persistent_bytes=draft.bytes + self.head_context.numel() * 2,
            greedy_only=False, max_depth=k, capabilities=Capabilities(short_drafts=True, batched_commit=True))

    def _taps(self, layers: dict, norm: torch.Tensor) -> list[torch.Tensor]:
        """The target's features at the draft's visits, then its final norm, as BF16 [N, H] rows."""
        return [*(layers[i].flatten(0, -2) for i in self.draft.features), norm.flatten(0, -2)]

    def ingest_prefill(self, req, ids, start, result) -> None:
        n = len(ids)
        taps = self._taps(result["features"], result["norm"])
        positions = torch.arange(start, start + n, device=self.context.device)
        self.draft.write_context(taps, positions, req.pages[start:start + n])
        self.head_context[req.slot] = self.draft.head_context([t[-1:] for t in taps])[0]
        req.proposer_state = True

    def start(self, req, token) -> None:
        """The pending token reaches `propose` with the round's inputs."""

    def ready(self, req) -> bool:
        return req.proposer_state is True

    @staticmethod
    def _history(req, pages: torch.Tensor) -> torch.Tensor:
        """The request's pages by position through the block: the committed prefix's, then the round's reservation.
        The same tensor the verification reads (`step_pages`): built here once a round, reused there."""
        held = req.reservation
        if held is None:
            return torch.cat([req.pages[:req.committed_len], pages])
        if held.full_pages is None:
            held.full_pages = torch.cat([req.pages, held.pages])
        return held.full_pages

    def propose(self, requests, pages, inputs, *, graphs, depth=None) -> torch.Tensor:
        depth = self.k if depth is None else depth
        batch, block = len(requests), self.draft.block
        device = self.context.device
        histories = [self._history(r, pages[i, :block]) for i, r in enumerate(requests)]
        lengths = [r.committed_len + block for r in requests]
        sizes = self.sizes.get(depth, ()) if graphs else ()
        if not sizes or batch > sizes[-1]:
            indptr = torch.tensor([0, *lengths], dtype=torch.int32, pin_memory=device.type == "cuda").cumsum_(0).to(device, non_blocking=True)
            indices = torch.cat([h[:n] for h, n in zip(histories, lengths)]).to(torch.int32)
            hidden = self.draft.forward(inputs.tokens, inputs.positions, pages[:, :block], indptr, indices)
            return self.draft.draft(hidden[:, :depth].contiguous(), self.head_context[inputs.slots.long()], inputs.tokens)
        size = graph_set.pick(sizes, batch)
        st = self.static[size, depth]
        # rows past the batch repeat the first request: they write its block's K/V again, to the same pages
        fill = [*range(batch), *[0] * (size - batch)]
        for name, value in (("tokens", inputs.tokens), ("positions", inputs.positions), ("pages", pages[:, :block]), ("slots", inputs.slots)):
            st[name].copy_(value[fill] if size > batch else value)
        # The page lists go request by request into the graph's buffer (device copies) and their offsets from pinned
        # memory: a pageable upload would wait for the GPU to drain the commit before the draft could be queued.
        offsets = torch.tensor([0, *(lengths[i] for i in fill)], dtype=torch.int32, pin_memory=device.type == "cuda").cumsum_(0)
        st["indptr"].copy_(offsets, non_blocking=True)
        at = 0
        for i in fill:
            st["indices"][at:at + lengths[i]].copy_(histories[i][:lengths[i]])
            at += lengths[i]
        self.graphs[size, depth].replay()
        return st["drafts"][:batch].clone()

    def accepted_batch(self, requests, emitted, features) -> None:
        """The rows a round consumed (the pending token, then the accepted drafts) are committed: their context K/V
        go to their pages and the last one's features to the request's head context. Their rows, positions and slots
        reach the device in one pinned upload, so nothing here waits for the GPU."""
        device = self.context.device
        width = features.norm.shape[1]
        rows, positions, last, slots, pages = [], [], [], [], []
        for i, (r, e) in enumerate(zip(requests, emitted)):
            n, start = len(e), r.committed_len - len(e)
            rows += range(i * width, i * width + n)
            positions += range(start, start + n)
            last.append(i * width + n - 1)
            slots.append(r.slot)
            pages.append(r.pages[start:start + n])
        t, b = len(rows), len(requests)
        meta = torch.tensor(rows + positions + last + slots, dtype=torch.int64, pin_memory=device.type == "cuda").to(device, non_blocking=True)
        taps = self._taps(features.layers, features.norm)
        self.draft.write_context(taps, meta[t:2 * t], pages[0] if b == 1 else torch.cat(pages), rows=meta[:t])
        self.head_context.index_copy_(0, meta[2 * t + b:], self.draft.head_context(taps, rows=meta[2 * t:2 * t + b]))

    def prepare_release(self, req, snapshot) -> None:
        """Nothing to attach: the draft context lives in the pages."""

    def release(self, req) -> None:
        req.proposer_state = None

    def capture_graphs(self, log=None, depths=None) -> None:
        """A graph per (verify depth, batch bucket), the verify graphs' pairs: the block's layers and the head's draws
        (run eagerly, their many small operators each pay a launch)."""
        device, draft = self.context.device, self.draft
        block, capacity = draft.block, draft.cos.shape[0]  # a request's pages: its context and the block
        for depth, buckets in sorted((depths or {}).items()):
            self.sizes[depth] = tuple(buckets)
            for batch in buckets:
                st = {"tokens": torch.zeros(batch, dtype=torch.int64, device=device), "positions": torch.zeros(batch, dtype=torch.int64, device=device),
                      "pages": torch.arange(batch * block, dtype=torch.int64, device=device).view(batch, block),
                      "indptr": torch.arange(0, (batch + 1) * block, block, dtype=torch.int32, device=device),
                      "indices": torch.zeros(batch * capacity, dtype=torch.int32, device=device),
                      "slots": torch.zeros(batch, dtype=torch.int64, device=device)}
                st["indices"][:batch * block] = torch.arange(batch * block, dtype=torch.int32, device=device)

                def body(st=st, depth=depth):
                    hidden = draft.forward(st["tokens"], st["positions"], st["pages"], st["indptr"], st["indices"])
                    context = self.head_context.index_select(0, st["slots"])
                    st["drafts"] = draft.draft(hidden[:, :depth].contiguous(), context, st["tokens"])
                self.static[batch, depth] = st
                self.graphs[batch, depth] = graph_set.capture(self.pool, body)
                if log:
                    log(f"  captured DSpark draft graph for batch {batch}, depth {depth}")
