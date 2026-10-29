# training/ - Handle learned pose model

Status: SCAFFOLD. Skeleton, FK, an untrained stub model, tests and a latency
benchmark. No data prep or training loop yet. Design rationale and data plan:
`docs/ml-research.md`.

Non-commercial notice: models trained on AMASS (or any dataset with a
non-commercial term) inherit that restriction. Do not ship weights as part of a
commercial product. Keep this notice in the top-level README.

## Files
- `skeleton.py`  canonical 22-joint SMPL-like skeleton, effector slots, mirror map.
- `fk.py`        differentiable PyTorch FK, rot6d / axis-angle helpers.
- `model.py`     `ProtoResLite` stub (4.1M params, residual on current pose).
- `benchmark.py` params, ONNX export, onnxruntime CPU latency.
- `tests/`       pytest, CPU only.

## Setup (Windows, CPU is enough for tests and the benchmark)
    python -m venv .venv
    .venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

For the RTX 3050 use the CUDA index in `requirements.txt` instead.

## Run
    python -m pytest -q
    python benchmark.py
