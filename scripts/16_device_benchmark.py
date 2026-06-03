"""
Script 62 — Honest on-device inference benchmark for Lite-TCN-SE v8b
======================================================================
Uses the EXACT model class (LiteTCNSE) with the v8b config:
  channels=[64,128,128,256], dilations=[1,2,4,8], se_reduction=2,
  hrv_features=12, kernel=3, dropout=0.3  → 319,138 parameters.

Random weights are used (latency depends on architecture/input, not weights).
No invented numbers: N/A where tooling is unavailable.
Energy is estimated (clearly labelled).
"""
from __future__ import annotations
import os, sys, time, csv, platform, subprocess, gc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from src.models.lite_tcn_se import LiteTCNSE

BASE = __import__('pathlib').Path(__file__).parent.parent
OUT  = BASE / "results/results_device_benchmark.csv"

print("=" * 62)
print("STEP 0 — Hardware identity")
print("=" * 62)
uname = subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout.strip()
arch  = platform.machine()
cpu   = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                       capture_output=True, text=True).stdout.strip()
pcpu  = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                       capture_output=True, text=True).stdout.strip()
lcpu  = subprocess.run(["sysctl", "-n", "hw.logicalcpu"],
                       capture_output=True, text=True).stdout.strip()
perf  = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                       capture_output=True, text=True).stdout.strip()
effi  = subprocess.run(["sysctl", "-n", "hw.perflevel1.physicalcpu"],
                       capture_output=True, text=True).stdout.strip()

print(f"  uname    : {uname}")
print(f"  machine  : {arch}")
print(f"  CPU      : {cpu}")
print(f"  P-cores  : {perf}   E-cores: {effi}   Physical total: {pcpu}")
print(f"  OS       : macOS {platform.mac_ver()[0]}")

if "arm" not in arch.lower() and "aarch64" not in arch.lower():
    print("\nSTOP: not an ARM device. Results would not be valid ARM measurements.")
    sys.exit(1)
print(f"\n  ✓ arm64 confirmed — proceeding.\n")

DEVICE_STR = (f"{cpu} | arm64 | macOS {platform.mac_ver()[0]} | "
              f"{pcpu} cores ({perf}P+{effi}E)")
N_PHYS = int(pcpu)

print("=" * 62)
print("STEP 1 — Model: LiteTCNSE (exact v8b config, random weights)")
print("=" * 62)
model = LiteTCNSE(
    input_channels=4,
    num_classes=2,
    channels_per_layer=[64, 128, 128, 256],
    dilation_schedule=[1, 2, 4, 8],
    kernel_size=3,
    dropout_rate=0.3,
    se_reduction=2,
    hrv_features=12,
).eval()

total_params = sum(p.numel() for p in model.parameters())
fp32_size_mb = total_params * 4 / 1024**2
print(f"  Parameters : {total_params:,}")
print(f"  FP32 size  : {fp32_size_mb:.2f} MB  (params × 4 bytes)")

# MAC count
print("\n  MAC count (ptflops):")
try:
    from ptflops import get_model_complexity_info
    class _Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x, torch.zeros(x.shape[0], 12))
    macs_str, _ = get_model_complexity_info(
        _Wrap(model), (4, 1920),
        as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"    {macs_str}")
except Exception as e:
    macs_str = "N/A"
    print(f"    failed: {e}")

print("\n" + "=" * 62)
print("STEP 2 — Exports")
print("=" * 62)

dummy_x   = torch.randn(1, 4, 1920)
dummy_hrv = torch.randn(1, 12)

# FP32 checkpoint
fp32_path = BASE / "results/lite_tcn_se_v8b_fp32.pt"
torch.save(model.state_dict(), str(fp32_path))
fp32_disk_mb = fp32_path.stat().st_size / 1e6
print(f"  FP32 .pt   : {fp32_disk_mb:.2f} MB  → {fp32_path.name}")

# ONNX
onnx_path = BASE / "results/lite_tcn_se_v8b.onnx"
ONNX_OK   = False
onnx_disk_mb = None
try:
    torch.onnx.export(
        model, (dummy_x, dummy_hrv), str(onnx_path),
        input_names=["signal", "hrv"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"signal": {0: "batch"}, "hrv": {0: "batch"}},
    )
    onnx_disk_mb = onnx_path.stat().st_size / 1e6
    print(f"  ONNX       : {onnx_disk_mb:.2f} MB  → {onnx_path.name}")
    ONNX_OK = True
except Exception as e:
    print(f"  ONNX FAILED: {e}")

# INT8 dynamic quantization
INT8_OK      = False
int8_disk_mb = None
int8_path    = BASE / "results/lite_tcn_se_v8b_int8.pt"
try:
    torch.backends.quantized.engine = "qnnpack"
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8)
    torch.save(model_int8.state_dict(), str(int8_path))
    int8_disk_mb = int8_path.stat().st_size / 1e6
    print(f"  INT8 .pt   : {int8_disk_mb:.2f} MB  → {int8_path.name}")
    INT8_OK = True
except Exception as e:
    print(f"  INT8 FAILED: {e}")

print("\n  TFLite FP16: N/A  (tflite_runtime not available on macOS)")
print("  TFLite INT8: N/A")

N_WARMUP = 30
N_ITERS  = 500

def timeit(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    gc.collect()
    t = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    a = np.array(t)
    return dict(mean=a.mean(), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), std=a.std())

def bench_pytorch(m, n_threads, mc_passes=1):
    torch.set_num_threads(n_threads)
    x   = torch.randn(1, 4, 1920)
    hrv = torch.randn(1, 12)
    if mc_passes == 1:
        m.eval()
        def fn():
            with torch.no_grad():
                m(x, hrv)
    else:
        m.train()   # keep dropout active
        def fn():
            with torch.no_grad():
                torch.stack([m(x, hrv) for _ in range(mc_passes)])
    r = timeit(fn)
    m.eval()
    return r

def bench_onnx(path, n_threads):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts,
                                providers=["CPUExecutionProvider"])
    x   = np.random.randn(1, 4, 1920).astype(np.float32)
    hrv = np.random.randn(1, 12).astype(np.float32)
    def fn(): sess.run(None, {"signal": x, "hrv": hrv})
    return timeit(fn)

print("\n" + "=" * 62)
print(f"STEP 3 — Benchmarks  (warmup={N_WARMUP}, timed={N_ITERS} iters)")
print("=" * 62)

R = {}

for label, n_t in [("1-thread", 1), (f"{N_PHYS}-thread", N_PHYS)]:
    print(f"\n  [PyTorch FP32, {label}]")
    r = bench_pytorch(model, n_t, mc_passes=1)
    R[f"fp32_{label}"] = r
    print(f"    mean={r['mean']:.2f} ms  median={r['median']:.2f}  "
          f"p95={r['p95']:.2f}  std={r['std']:.2f}")

for label, n_t in [("1-thread", 1), (f"{N_PHYS}-thread", N_PHYS)]:
    print(f"\n  [PyTorch MC-Dropout N=30, {label}]")
    r = bench_pytorch(model, n_t, mc_passes=30)
    R[f"mc30_{label}"] = r
    print(f"    mean={r['mean']:.2f} ms  median={r['median']:.2f}  "
          f"p95={r['p95']:.2f}  std={r['std']:.2f}")

if ONNX_OK:
    for label, n_t in [("1-thread", 1), (f"{N_PHYS}-thread", N_PHYS)]:
        print(f"\n  [ONNX Runtime, {label}]")
        r = bench_onnx(onnx_path, n_t)
        R[f"onnx_{label}"] = r
        print(f"    mean={r['mean']:.2f} ms  median={r['median']:.2f}  "
              f"p95={r['p95']:.2f}  std={r['std']:.2f}")

if INT8_OK:
    for label, n_t in [("1-thread", 1), (f"{N_PHYS}-thread", N_PHYS)]:
        print(f"\n  [PyTorch INT8, {label}]")
        r = bench_pytorch(model_int8, n_t, mc_passes=1)
        R[f"int8_{label}"] = r
        print(f"    mean={r['mean']:.2f} ms  median={r['median']:.2f}  "
              f"p95={r['p95']:.2f}  std={r['std']:.2f}")

print("\n" + "=" * 62)
print("STEP 5 — Energy (ESTIMATED — no hardware power meter)")
print("=" * 62)
# Apple M4 Pro single P-core sustained CPU load: ~3 W (conservative)
# Source: Apple Silicon power characterisation literature & Asahi Linux measurements
POWER_1T_W = 3.0
print(f"  Assumed single-core active power: {POWER_1T_W} W")
print("  Energy = median_latency_s × assumed_power_W  → millijoules")
print("  Label: ESTIMATED")

def energy_mj(median_ms): return median_ms * 1e-3 * POWER_1T_W * 1e3

print("\n" + "=" * 62)
print("STEP 6 — Summary")
print("=" * 62)

def v(key, field):
    r = R.get(key)
    return f"{r[field]:.2f}" if r else "N/A"

def e(key):
    r = R.get(key)
    return f"{energy_mj(r['median']):.3f}" if r else "N/A"

rows = [
    {
        "variant": "PyTorch FP32",
        "params": f"{total_params:,}",
        "MACs": macs_str,
        "size_MB": f"{fp32_disk_mb:.2f}",
        "lat_med_1t_ms":  v("fp32_1-thread",       "median"),
        "lat_p95_1t_ms":  v("fp32_1-thread",       "p95"),
        "lat_med_mt_ms":  v(f"fp32_{N_PHYS}-thread","median"),
        "MC30_med_1t_ms": v("mc30_1-thread",       "median"),
        "MC30_p95_1t_ms": v("mc30_1-thread",       "p95"),
        "MC30_med_mt_ms": v(f"mc30_{N_PHYS}-thread","median"),
        "energy_mJ":      e("fp32_1-thread"),
        "energy_type":    "estimated",
        "assumed_W":      str(POWER_1T_W),
        "device":         DEVICE_STR,
    },
    {
        "variant": "ONNX Runtime FP32",
        "params": f"{total_params:,}",
        "MACs": macs_str,
        "size_MB": f"{onnx_disk_mb:.2f}" if onnx_disk_mb else "N/A",
        "lat_med_1t_ms":  v("onnx_1-thread",        "median"),
        "lat_p95_1t_ms":  v("onnx_1-thread",        "p95"),
        "lat_med_mt_ms":  v(f"onnx_{N_PHYS}-thread","median"),
        "MC30_med_1t_ms": "N/A",
        "MC30_p95_1t_ms": "N/A",
        "MC30_med_mt_ms": "N/A",
        "energy_mJ":      e("onnx_1-thread"),
        "energy_type":    "estimated",
        "assumed_W":      str(POWER_1T_W),
        "device":         DEVICE_STR,
    },
    {
        "variant": "PyTorch INT8 (dynamic)",
        "params": f"{total_params:,}",
        "MACs": "N/A",
        "size_MB": f"{int8_disk_mb:.2f}" if int8_disk_mb else "N/A",
        "lat_med_1t_ms":  v("int8_1-thread",        "median"),
        "lat_p95_1t_ms":  v("int8_1-thread",        "p95"),
        "lat_med_mt_ms":  v(f"int8_{N_PHYS}-thread","median"),
        "MC30_med_1t_ms": "N/A",
        "MC30_p95_1t_ms": "N/A",
        "MC30_med_mt_ms": "N/A",
        "energy_mJ":      e("int8_1-thread"),
        "energy_type":    "estimated",
        "assumed_W":      str(POWER_1T_W),
        "device":         DEVICE_STR,
    },
    {
        "variant": "TFLite FP16", "params": "N/A", "MACs": "N/A",
        "size_MB": "N/A", "lat_med_1t_ms": "N/A", "lat_p95_1t_ms": "N/A",
        "lat_med_mt_ms": "N/A", "MC30_med_1t_ms": "N/A",
        "MC30_p95_1t_ms": "N/A", "MC30_med_mt_ms": "N/A",
        "energy_mJ": "N/A", "energy_type": "N/A", "assumed_W": "N/A",
        "device": DEVICE_STR,
    },
    {
        "variant": "TFLite INT8", "params": "N/A", "MACs": "N/A",
        "size_MB": "N/A", "lat_med_1t_ms": "N/A", "lat_p95_1t_ms": "N/A",
        "lat_med_mt_ms": "N/A", "MC30_med_1t_ms": "N/A",
        "MC30_p95_1t_ms": "N/A", "MC30_med_mt_ms": "N/A",
        "energy_mJ": "N/A", "energy_type": "N/A", "assumed_W": "N/A",
        "device": DEVICE_STR,
    },
]

# Print
cols  = ["variant","params","size_MB","lat_med_1t_ms","lat_p95_1t_ms",
         "lat_med_mt_ms","MC30_med_1t_ms","energy_mJ","energy_type"]
heads = ["Variant","Params","Size(MB)","Lat-med 1T","Lat-p95 1T",
         f"Lat-med {N_PHYS}T","MC30-med 1T","Energy(mJ)","Meas/Est"]
widths = [24, 10, 9, 11, 11, 11, 12, 11, 10]
sep = "  ".join("-"*w for w in widths)
fmt = "  ".join(f"{{:<{w}}}" for w in widths)
print()
print(fmt.format(*heads))
print(sep)
for row in rows:
    print(fmt.format(*[row.get(c,"N/A") for c in cols]))

# CSV
with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  CSV saved → {OUT}")

fp32_1t  = R.get("fp32_1-thread",  {}).get("median", 0)
fp32_mt  = R.get(f"fp32_{N_PHYS}-thread", {}).get("median", 0)
mc30_1t  = R.get("mc30_1-thread",  {}).get("median", 0)
mc30_mt  = R.get(f"mc30_{N_PHYS}-thread", {}).get("median", 0)
onnx_1t  = R.get("onnx_1-thread",  {}).get("median", 0)
onnx_mt  = R.get(f"onnx_{N_PHYS}-thread", {}).get("median", 0)
int8_1t  = R.get("int8_1-thread",  {}).get("median", 0)

print("\n" + "=" * 62)
print("PAPER NOTE  (paste directly)")
print("=" * 62)
print(f"""
All latency figures were measured on an Apple M4 Pro ({pcpu}-core ARM64,
macOS {platform.mac_ver()[0]}) using PyTorch {torch.__version__} CPU execution
(30 warmup passes, 500 timed passes, input [1, 4, 1920] + 12-dim HRV vector).
Single-thread FP32 median latency: {fp32_1t:.1f} ms (p95: {R.get('fp32_1-thread',{}).get('p95',0):.1f} ms);
{N_PHYS}-thread FP32 median: {fp32_mt:.1f} ms.
Monte Carlo Dropout (N=30 passes, single thread) median: {mc30_1t:.1f} ms,
which corresponds to {mc30_1t/30:.1f} ms per stochastic pass.
ONNX Runtime (opset 17) single-thread median: {onnx_1t:.1f} ms;
{N_PHYS}-thread: {onnx_mt:.1f} ms.
PyTorch dynamic INT8 (qnnpack) single-thread median: {int8_1t:.1f} ms.
Energy per inference is an estimate: median latency (s) × {POWER_1T_W} W
assumed single-core active power (no hardware power meter was used;
the {POWER_1T_W} W figure is a conservative single-P-core load estimate for
Apple M-series silicon). TFLite variants are marked N/A because
tflite_runtime is unavailable on macOS in this environment.
""")
