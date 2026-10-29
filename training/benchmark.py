"""Parameter count, ONNX size and CPU latency of the (untrained) stub model.

    python benchmark.py [--runs 500] [--out build/stub.onnx]

Latency here is for RANDOM weights; speed does not depend on weight values,
but accuracy numbers obviously do, and this script reports none.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import torch

from model import EFF_DIM, POSE_DIM, ProtoResLite, count_params
from skeleton import NUM_EFFECTORS


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def bench(fn, runs, warmup=50):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.mean(ts), percentile(ts, 0.5), percentile(ts, 0.99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--out", default=os.path.join("build", "stub.onnx"))
    a = ap.parse_args()

    torch.manual_seed(0)
    m = ProtoResLite().eval()
    n = count_params(m)
    print(f"params: {n:,}  fp32 {n*4/1e6:.1f} MB  fp16 {n*2/1e6:.1f} MB")
    eff = torch.randn(1, NUM_EFFECTORS, EFF_DIM)
    eff[..., 0] = 1.0
    pose = torch.randn(1, POSE_DIM)

    with torch.inference_mode():
        mean, p50, p99 = bench(lambda: m(eff, pose), a.runs)
    print(f"torch cpu ({torch.get_num_threads()} threads) batch1: mean {mean:.3f} ms p50 {p50:.3f} p99 {p99:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(
        m, (eff, pose), a.out, input_names=["effectors", "cur_pose"], output_names=["pose"],
        opset_version=17, dynamo=False, dynamic_axes={"effectors": {0: "b"}, "cur_pose": {0: "b"}, "pose": {0: "b"}},
    )
    print(f"onnx file: {os.path.getsize(a.out)/1e6:.2f} MB -> {a.out}")

    import onnxruntime as ort

    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
        feed = {"effectors": eff.numpy(), "cur_pose": pose.numpy()}
        mean, p50, p99 = bench(lambda: sess.run(None, feed), a.runs)
        print(f"onnxruntime {ort.__version__} cpu threads={threads} batch1: mean {mean:.3f} ms p50 {p50:.3f} p99 {p99:.3f}")
        if threads == 1:
            ref = m(eff, pose).detach().numpy()
            out = sess.run(None, feed)[0]
            print(f"  max |torch - onnx| = {np.abs(ref - out).max():.2e}")


if __name__ == "__main__":
    main()
