# dataloader_autotune.py
import os, time, math, itertools, statistics as stats
import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
def _dl(device, dataset, batch_size, num_workers, pin_memory, prefetch_factor, persistent_workers, collate_fn=None, mp_ctx=None):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        persistent_workers=persistent_workers and (num_workers > 0),
        drop_last=True,  # steadier throughput
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if mp_ctx is not None:
        kwargs["multiprocessing_context"] = mp_ctx
    return DataLoader(**kwargs)

@torch.no_grad()
def benchmark_loader(model=None, device=None, dataset=None, collate_fn=None,
                     warmup_batches=20, measure_batches=100,
                     batch_sizes=(2048, 4096, 8192, 16384),
                     worker_choices=(0, 2, 4, 8, 12, 16),
                     pin_memory_choices=(True, False),
                     prefetch_choices=(2, 4),
                     persistent=True,
                     mp_ctx=None):

    assert dataset is not None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    # Simple consumer that mimics your training step (uses model if provided)
    def _consume(batch):
        # batch may be (u,i,r) for MF; move to device non_blocking if possible
        if isinstance(batch, (list, tuple)):
            moved = []
            for x in batch:
                if torch.is_tensor(x):
                    moved.append(x.to(device, non_blocking=True))
                else:
                    moved.append(x)
            if model is not None:
                u,i,*rest = moved
                _ = model(u, i)  # forward only; no backward for loader speed
            return
        elif torch.is_tensor(batch):
            _ = batch.to(device, non_blocking=True)

    for bs, nw, pm in itertools.product(batch_sizes, worker_choices, pin_memory_choices):
        pf_list = prefetch_choices if nw > 0 else [None]
        for pf in pf_list:
            try:
                loader = _dl(device, dataset, bs, nw, pm, pf, persistent, collate_fn, mp_ctx)
            except Exception as e:
                results.append(dict(
                    batch_size=bs, num_workers=nw, pin_memory=pm, prefetch=pf,
                    ttfb_s=math.inf, samples_per_s=0.0, note=f"init_fail: {e}"
                ))
                continue

            # Warmup
            it = iter(loader)
            try:
                t0 = time.perf_counter()
                first = next(it)
                _consume(first)
                ttfb = time.perf_counter() - t0
            except StopIteration:
                results.append(dict(
                    batch_size=bs, num_workers=nw, pin_memory=pm, prefetch=pf,
                    ttfb_s=math.inf, samples_per_s=0.0, note="empty_dataset"
                ))
                continue

            for _ in range(max(0, warmup_batches - 1)):
                try:
                    _consume(next(it))
                except StopIteration:
                    break

            # Measure
            times = []
            n_batches = 0
            n_samples = 0
            t_start = time.perf_counter()
            while n_batches < measure_batches:
                try:
                    b = next(it)
                except StopIteration:
                    break
                _t = time.perf_counter()
                _consume(b)
                times.append(time.perf_counter() - _t)
                n_batches += 1
                n_samples += bs
            total_s = time.perf_counter() - t_start
            samples_per_s = n_samples / total_s if total_s > 0 else 0.0

            results.append(dict(
                batch_size=bs,
                num_workers=nw,
                pin_memory=pm,
                prefetch=pf,
                ttfb_s=ttfb,
                median_batch_copy_s=stats.median(times) if times else float("nan"),
                samples_per_s=round(samples_per_s, 2),
                note=""
            ))
    # Sort: higher throughput, then lower TTFB
    results.sort(key=lambda d: (-d["samples_per_s"], d["ttfb_s"]))
    return results

def print_top(results, top=10):
    print(f"\nTop {top} configs by throughput (samples/s):")
    for r in results[:top]:
        print(
            f"bs={r['batch_size']:>6} | workers={r['num_workers']:>2} | pin={str(r['pin_memory'])[0]} "
            f"| pf={str(r['prefetch']):>4} | TTFB={r['ttfb_s']:.3f}s | median_copy={r['median_batch_copy_s']:.4f}s "
            f"| {r['samples_per_s']} samp/s {r['note']}"
        )
