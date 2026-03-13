import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _first_floating_dtype(model: torch.nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _output_to_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.ndim > 2:
            output = output.reshape(output.shape[0], -1)
        return output

    if isinstance(output, (list, tuple)) and output:
        for item in output:
            try:
                return _output_to_logits(item)
            except ValueError:
                continue

    if isinstance(output, dict) and output:
        for _, value in output.items():
            try:
                return _output_to_logits(value)
            except ValueError:
                continue

    raise ValueError("Model output is not a supported tensor/list/dict tensor structure.")


def _model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(model, tmp_path)
        return tmp_path.stat().st_size / (1024 * 1024)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def evaluate_model(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    num_samples: int = 1000,
    reference_model: torch.nn.Module | None = None,
) -> Dict[str, float]:
    model.eval()
    if reference_model is not None:
        reference_model.eval()

    model_dtype = _first_floating_dtype(model)
    ref_dtype = _first_floating_dtype(reference_model) if reference_model is not None else model_dtype
    x = torch.randn((num_samples, *input_shape), dtype=model_dtype)

    with torch.no_grad():
        output = _output_to_logits(model(x[:1]))

    num_classes = int(output.shape[1])

    if reference_model is not None:
        with torch.no_grad():
            ref_input = x.to(ref_dtype)
            ref_logits = _output_to_logits(reference_model(ref_input))
            labels = torch.argmax(ref_logits, dim=1)
    else:
        labels = torch.randint(0, num_classes, (num_samples,))

    batch_size = 16
    preds: list[torch.Tensor] = []
    infer_time_start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            xb = x[i : i + batch_size]
            logits = _output_to_logits(model(xb))
            preds.append(torch.argmax(logits, dim=1))
    infer_time_sec = time.perf_counter() - infer_time_start

    predictions = torch.cat(preds, dim=0)
    accuracy = float((predictions == labels).float().mean().item() * 100.0)

    latency_probe = min(200, num_samples)
    latencies_ms: list[float] = []
    with torch.no_grad():
        for i in range(latency_probe):
            sample = x[i : i + 1]
            start = time.perf_counter()
            _ = _output_to_logits(model(sample))
            latencies_ms.append((time.perf_counter() - start) * 1000)

    throughput = float(num_samples / max(infer_time_sec, 1e-9))

    parameter_count = int(sum(p.numel() for p in model.parameters()))

    return {
        "accuracy": accuracy,
        "model_size_mb": _model_size_mb(model),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_samples_per_sec": throughput,
        "parameter_count": parameter_count,
    }


def compare_models(original: Dict[str, float], compressed: Dict[str, float]) -> Dict[str, float]:
    orig_size = float(original["model_size_mb"])
    comp_size = float(compressed["model_size_mb"])
    orig_acc = float(original["accuracy"])
    comp_acc = float(compressed["accuracy"])
    orig_lat = float(original["latency_p95_ms"])
    comp_lat = float(compressed["latency_p95_ms"])

    return {
        "original_size_mb": orig_size,
        "compressed_size_mb": comp_size,
        "size_reduction_percent": ((orig_size - comp_size) / max(orig_size, 1e-9)) * 100.0,
        "original_accuracy": orig_acc,
        "compressed_accuracy": comp_acc,
        "accuracy_difference": comp_acc - orig_acc,
        "original_latency_ms": orig_lat,
        "compressed_latency_ms": comp_lat,
        "speedup": orig_lat / max(comp_lat, 1e-9),
        "compression_ratio": orig_size / max(comp_size, 1e-9),
    }
