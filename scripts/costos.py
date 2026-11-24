"""Costo de cada variante independiente del hardware (MACs, FLOPs) y de inferencia (latencia, memoria, disco).

Responde a dos observaciones de revisión: el tiempo por época depende de la GPU y de la implementación
(las cabezas se calculan en un bucle), y el artículo no reportaba métricas de inferencia.

Para cada variante carga el ``best.pt`` de la semilla 0 en CIFAR-10 (no reentrena nada) y mide:

- MACs por imagen en inferencia y FLOPs de entrenamiento por época (forward + backward de un lote,
  escalado a 45 000 imágenes), con ``torch.utils.flop_counter.FlopCounterMode``. Cuenta matmul, bmm,
  convoluciones (también depthwise; el backward agrupado con una fórmula propia, ver
  ``conv_backward_flop``) y SDPA; NO cuenta operaciones elemento a elemento (softmax, tanh,
  GELU, normalizaciones). MACs = FLOPs / 2.
- Memoria pico en GPU de un paso de entrenamiento con el protocolo real (lote 256, fp32, AdamW).
- Pico de activaciones en inferencia con lote 1: bytes de los tensores intermedios vivos durante el forward
  (en CPU, con un ``TorchDispatchMode``; no incluye parámetros ni espacios de trabajo del asignador).
- Latencia en CPU con lote 1 (1 hilo y todos los hilos; mediana de N repeticiones tras calentamiento).
- Tamaño en disco del ``state_dict`` fp32.

Uso (desde ham/):  uv run --offline python scripts/costos.py [--reps 300] [--out results/costos.json]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode, conv_flop_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ham.utils.io import load_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGES = 45_000  # división 45k/5k/10k de CIFAR-10

MAIN = ["dot", "dot_wide", "ham_concat", "ham_sum", "dot_local", "dot_local_k5", "dot_wide_local_k5",
        "ham_concat_local_k5"]
BASELINES = ["localvit", "spt_lsa", "convit", "conv_stem", "resnet"]
SCALES = ["d96_l4", "d192_l4", "d48_l8", "d96_l8", "d192_l8"]
VARIANTS = MAIN + BASELINES + [f"{b}_{s}" for s in SCALES for b in ("dot", "dot_local_k5")]


def run_dir(name: str) -> Path:
    return ROOT / "runs" / "cifar10" / f"cifar10_{name}" / "seed0"


def conv_backward_flop(grad_out_shape, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed,
                       _output_padding, _groups, output_mask, out_shape=None, **kwargs) -> int:
    """Backward de una convolución: grad_input y grad_weight cuestan cada uno lo mismo que el forward.

    Sustituye a la fórmula de torch (2.14), que ignora ``groups`` en el backward y cuenta una convolución
    depthwise de C canales como si fuera densa (C veces más). ``conv_flop_count`` con ``w_shape`` =
    (c_out, c_in/groups, k, k) sí es correcta para convoluciones agrupadas.
    """
    fwd = conv_flop_count(x_shape, w_shape, grad_out_shape, transposed=transposed)
    return fwd * (int(bool(output_mask[0])) + int(bool(output_mask[1])))


def flops(model: nn.Module, x: torch.Tensor, backward: bool = False) -> int:
    with FlopCounterMode(display=False, custom_mapping={torch.ops.aten.convolution_backward: conv_backward_flop}) as fc:
        out = model(x)
        if backward:
            out.sum().backward()
    return int(fc.get_total_flops())


def train_step_memory(model: nn.Module, cfg, device) -> float:
    """Pico de memoria (MB) de un paso de entrenamiento completo con AdamW en fp32."""
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bs = cfg.train.batch_size
    x = torch.randn(bs, cfg.model.num_channels, cfg.model.image_size, cfg.model.image_size, device=device)
    y = torch.randint(0, cfg.model.num_classes, (bs,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(2):  # el primer paso crea los estados de AdamW
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    del opt
    model.zero_grad(set_to_none=True)
    return peak / 2**20


class _LiveTensors(TorchDispatchMode):
    """Cuenta los bytes de los tensores intermedios vivos durante un forward y guarda el pico.

    Cada almacenamiento nuevo (no una vista) que devuelve una operación suma sus bytes; al liberarse,
    los resta. No depende del dispositivo ni del asignador de memoria (a diferencia de
    ``torch.cuda.max_memory_allocated``, que incluye espacios de trabajo de cuDNN/cuBLAS y redondeos).
    """

    def __init__(self):
        super().__init__()
        self.live = 0
        self.peak = 0
        self.seen = set()

    def _free(self, key, n):
        self.live -= n
        self.seen.discard(key)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in torch.utils._pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor):
                st = t.untyped_storage()
                key = st.data_ptr()
                if key and key not in self.seen:
                    self.seen.add(key)
                    n = st.nbytes()
                    self.live += n
                    weakref.finalize(st, self._free, key, n)
        self.peak = max(self.peak, self.live)
        return out


def activation_peak_kb(model: nn.Module, x: torch.Tensor) -> float:
    """Pico de memoria de activaciones (KB) de un forward en inferencia, sin contar parámetros ni la entrada."""
    model = model.cpu().eval()
    with torch.inference_mode():
        model(x)  # calentamiento (cachés perezosas)
        mode = _LiveTensors()
        with mode:
            out = model(x)
        del out
    return mode.peak / 1024


def cpu_latency_ms(model: nn.Module, x: torch.Tensor, threads: int, reps: int, warmup: int = 50) -> dict:
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = x.cpu()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    q = statistics.quantiles(times, n=4)
    return {"median": statistics.median(times), "q1": q[0], "q3": q[2]}


def disk_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024


def machine() -> dict:
    cpu = platform.processor()
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}).stdout
        cpu = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")), cpu)
    except OSError:
        pass
    return {"cpu": cpu, "cpu_threads": os.cpu_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def measure(name: str, reps: int, device) -> dict:
    cfg, model = load_run(run_dir(name))
    c, s = cfg.model.num_channels, cfg.model.image_size
    x1 = torch.randn(1, c, s, s)
    n_params = sum(p.numel() for p in model.parameters())
    model.eval()
    inf_flops = flops(model, x1)
    model.train()
    xb = torch.randn(8, c, s, s)
    train_flops_img = flops(model, xb, backward=True) / 8
    model.zero_grad(set_to_none=True)
    r = {
        "params": n_params,
        "macs_img": inf_flops / 2,
        "train_flops_epoch": train_flops_img * TRAIN_IMAGES,
        "disk_kb": disk_kb(model),
        "act_peak_kb": activation_peak_kb(model, x1),
        "cpu_ms_1thread": cpu_latency_ms(model, x1, 1, reps),
        "cpu_ms_allthreads": cpu_latency_ms(model, x1, os.cpu_count(), reps),
    }
    if device.type == "cuda":
        r["train_mem_mb"] = train_step_memory(model, cfg, device)
        model.cpu()
        torch.cuda.empty_cache()
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "costos.json")
    p.add_argument("--only", nargs="*", help="medir sólo estas variantes")
    a = p.parse_args(argv)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"machine": machine(), "reps": a.reps, "train_images": TRAIN_IMAGES, "variants": {}}
    for name in a.only or VARIANTS:
        if not (run_dir(name) / "best.pt").exists():
            print(f"[omitida] {name}: no hay best.pt", flush=True)
            continue
        r = measure(name, a.reps, device)
        res["variants"][name] = r
        print(f"{name:28s} {r['params']:>9,d} p  {r['macs_img'] / 1e6:7.2f} MMACs  "
              f"{r['train_flops_epoch'] / 1e12:7.2f} TFLOPs/ép  {r.get('train_mem_mb', 0):7.0f} MB  "
              f"{r['cpu_ms_1thread']['median']:6.2f} ms(1h)  {r['cpu_ms_allthreads']['median']:6.2f} ms(todos)  "
              f"{r.get('act_peak_kb', 0):7.0f} KB act  {r['disk_kb']:6.0f} KB", flush=True)
    torch.set_num_threads(os.cpu_count())
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
