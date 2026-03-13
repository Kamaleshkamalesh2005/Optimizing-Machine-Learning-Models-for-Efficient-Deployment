import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.evaluate import compare_models, evaluate_model


class StudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class CompressionEngine:
    def __init__(self, upload_path: Path, compressed_dir: Path, results_path: Path):
        self.upload_path = upload_path
        self.compressed_dir = compressed_dir
        self.results_path = results_path

        self.original_model: nn.Module | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.original_metrics: Dict[str, float] | None = None
        self.baseline_labels_ref_model: nn.Module | None = None

        self.candidates: List[Tuple[str, nn.Module, Dict[str, float], Dict[str, float]]] = []
        self.technique_results: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)

    @staticmethod
    def _layer_count(model: nn.Module) -> int:
        return sum(1 for _ in model.modules()) - 1

    @staticmethod
    def _parameter_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def _first_floating_dtype(model: nn.Module) -> torch.dtype:
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype
        return torch.float32

    @staticmethod
    def _to_logits(output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            if output.ndim == 1:
                output = output.unsqueeze(0)
            if output.ndim > 2:
                output = output.reshape(output.shape[0], -1)
            return output

        if isinstance(output, (list, tuple)) and output:
            for item in output:
                try:
                    return CompressionEngine._to_logits(item)
                except ValueError:
                    continue

        if isinstance(output, dict) and output:
            for _, value in output.items():
                try:
                    return CompressionEngine._to_logits(value)
                except ValueError:
                    continue

        raise ValueError("Unsupported model output format for compression engine.")

    @staticmethod
    def _infer_input_shape(model: nn.Module) -> tuple[int, ...]:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                return (module.in_channels, 224, 224)
            if isinstance(module, nn.Linear):
                return (module.in_features,)
        return (3, 224, 224)

    def load_original_model(self) -> nn.Module:
        try:
            payload = torch.load(self.upload_path, map_location="cpu")
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "unknown")
            raise ValueError(
                f"Missing dependency '{missing}' required by this checkpoint. "
                f"Install it in the server environment, or export/upload a plain torch.nn.Module checkpoint."
            ) from exc
        if isinstance(payload, nn.Module):
            model = payload
        elif isinstance(payload, dict) and isinstance(payload.get("model"), nn.Module):
            model = payload["model"]
        elif isinstance(payload, dict) and ("state_dict" in payload or all(torch.is_tensor(v) for v in payload.values())):
            raise ValueError(
                "State-dict-only checkpoint detected. Upload a full serialized torch.nn.Module, "
                "or adapt this service with the model architecture code."
            )
        else:
            raise ValueError(
                "Uploaded file must contain a serialized torch.nn.Module object. "
                "State-dict-only checkpoints are not supported."
            )

        # Run on CPU in fp32 for stable inference and compression operations.
        model = model.to("cpu")
        if self._first_floating_dtype(model) != torch.float32:
            model = model.float()

        model.eval()
        self.original_model = model
        self.input_shape = self._infer_input_shape(model)
        self.baseline_labels_ref_model = copy.deepcopy(model).eval()

        self.original_metrics = evaluate_model(
            model=model,
            input_shape=self.input_shape,
            reference_model=self.baseline_labels_ref_model,
        )
        self.candidates = [("original", copy.deepcopy(model).eval(), self.original_metrics, {"kept": 1.0})]
        return model

    def model_info(self) -> Dict[str, float | list[int]]:
        if self.original_model is None or self.input_shape is None:
            self.load_original_model()
        assert self.original_model is not None
        assert self.input_shape is not None

        return {
            "size_mb": self._file_size_mb(self.upload_path),
            "parameter_count": self._parameter_count(self.original_model),
            "layer_count": self._layer_count(self.original_model),
            "input_shape": list(self.input_shape),
        }

    def _evaluate_candidate(self, model: nn.Module, num_samples: int = 1000) -> Dict[str, float]:
        assert self.input_shape is not None
        assert self.baseline_labels_ref_model is not None
        return evaluate_model(
            model=model,
            input_shape=self.input_shape,
            num_samples=num_samples,
            reference_model=self.baseline_labels_ref_model,
        )

    def run_dynamic_quantization(self) -> Dict[str, float]:
        assert self.original_model is not None
        original = copy.deepcopy(self.original_model).eval()
        before_size = self.original_metrics["model_size_mb"] if self.original_metrics else 0.0

        try:
            quantized = torch.quantization.quantize_dynamic(
                original,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8,
            )
        except Exception:
            quantized = torch.quantization.quantize_dynamic(
                original,
                {nn.Linear},
                dtype=torch.qint8,
            )

        metrics = self._evaluate_candidate(quantized)
        after_size = metrics["model_size_mb"]
        reduction = ((before_size - after_size) / max(before_size, 1e-9)) * 100.0

        self.technique_results["quantization"] = {
            "size_before_mb": before_size,
            "size_after_mb": after_size,
            "size_reduction_percent": reduction,
            "accuracy": metrics["accuracy"],
        }
        self.candidates.append(("quantized", quantized, metrics, self.technique_results["quantization"]))
        return self.technique_results["quantization"]

    def run_pruning(self) -> Dict[str, float]:
        assert self.original_model is not None
        assert self.original_metrics is not None

        target_acc = self.original_metrics["accuracy"] - 1.0
        best_model: nn.Module | None = None
        best_metrics: Dict[str, float] | None = None
        best_sparsity = 0.0

        for sparsity_step in range(1, 10):
            sparsity = sparsity_step / 10.0
            candidate = copy.deepcopy(self.original_model).eval()
            for module in candidate.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name="weight", amount=sparsity)
                    prune.remove(module, "weight")

            # Use a fast subset for sparsity search, then evaluate final winner with full samples.
            metrics = self._evaluate_candidate(candidate, num_samples=200)
            if metrics["accuracy"] >= target_acc:
                best_model = candidate
                best_metrics = metrics
                best_sparsity = sparsity
            else:
                break

        if best_model is None:
            best_model = copy.deepcopy(self.original_model).eval()
            best_metrics = self._evaluate_candidate(best_model, num_samples=1000)
            best_sparsity = 0.0
        else:
            best_metrics = self._evaluate_candidate(best_model, num_samples=1000)

        self.technique_results["pruning"] = {
            "best_sparsity_percent": best_sparsity * 100.0,
            "accuracy": best_metrics["accuracy"],
        }
        self.candidates.append(("pruned", best_model, best_metrics, self.technique_results["pruning"]))
        return self.technique_results["pruning"]

    @staticmethod
    def _extract_linear_layout(model: nn.Module) -> Tuple[int, List[int], int]:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(linears) < 2:
            raise ValueError("Teacher model does not expose enough Linear layers for distillation auto-build.")

        input_dim = int(linears[0].in_features)
        output_dim = int(linears[-1].out_features)
        hidden = [int(l.out_features) for l in linears[:-1]]
        return input_dim, hidden, output_dim

    def run_distillation(self) -> Dict[str, float]:
        assert self.original_model is not None
        assert self.original_metrics is not None
        assert self.input_shape is not None

        teacher = copy.deepcopy(self.original_model).eval()
        target_acc = self.original_metrics["accuracy"] - 1.0

        try:
            input_dim, teacher_hidden, output_dim = self._extract_linear_layout(teacher)
            reduced_hidden = [max(8, h // 2) for h in teacher_hidden]
            half_layer_count = max(1, len(reduced_hidden) // 2)
            student_hidden = reduced_hidden[:half_layer_count]
            student = StudentMLP(input_dim=input_dim, hidden_dims=student_hidden, output_dim=output_dim)
            flatten_required = True
        except Exception:
            # Distillation auto-build is only reliable for MLP-style classifiers with linear heads.
            self.technique_results["distillation"] = {
                "student_size_percent_of_teacher": 100.0,
                "accuracy": self.original_metrics["accuracy"],
                "kept": 0.0,
                "skipped": 1.0,
            }
            return self.technique_results["distillation"]

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        kl_div = nn.KLDivLoss(reduction="batchmean")
        ce = nn.CrossEntropyLoss()
        temperature = 4.0

        batch_size = 32
        train_samples = 1024

        for _ in range(20):
            teacher_dtype = self._first_floating_dtype(teacher)
            x = torch.randn((train_samples, *self.input_shape), dtype=teacher_dtype)
            with torch.no_grad():
                teacher_logits = self._to_logits(teacher(x))
                hard_labels = torch.argmax(teacher_logits, dim=1)

            for i in range(0, train_samples, batch_size):
                xb = x[i : i + batch_size]
                t_logits = teacher_logits[i : i + batch_size]
                yb = hard_labels[i : i + batch_size]

                optimizer.zero_grad()
                s_logits = student(xb.float() if flatten_required else xb.float())

                kd = kl_div(
                    nn.functional.log_softmax(s_logits / temperature, dim=1),
                    nn.functional.softmax(t_logits / temperature, dim=1),
                ) * (temperature**2)
                ce_loss = ce(s_logits, yb)
                loss = 0.7 * kd + 0.3 * ce_loss
                loss.backward()
                optimizer.step()

        student.eval()
        metrics = self._evaluate_candidate(student)

        kept = 1.0
        if metrics["accuracy"] >= target_acc:
            self.candidates.append(
                (
                    "distilled",
                    student,
                    metrics,
                    {
                        "student_size_percent_of_teacher": (
                            metrics["model_size_mb"] / max(self.original_metrics["model_size_mb"], 1e-9)
                        )
                        * 100.0,
                        "accuracy": metrics["accuracy"],
                    },
                )
            )
            kept = 1.0

        self.technique_results["distillation"] = {
            "student_size_percent_of_teacher": (
                metrics["model_size_mb"] / max(self.original_metrics["model_size_mb"], 1e-9)
            )
            * 100.0,
            "accuracy": metrics["accuracy"],
            "kept": kept,
            "skipped": 0.0,
        }
        return self.technique_results["distillation"]

    @staticmethod
    def _benchmark_onnx(onnx_path: Path, input_shape: tuple[int, ...], runs: int = 500) -> float:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        x = np.random.randn(1, *input_shape).astype(np.float32)

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            session.run(None, {input_name: x})
            latencies.append((time.perf_counter() - start) * 1000)
        return float(np.percentile(latencies, 95))

    @staticmethod
    def _benchmark_torch(model: nn.Module, input_shape: tuple[int, ...], runs: int = 500) -> float:
        x = torch.randn((1, *input_shape), dtype=torch.float32)
        model.eval()
        latencies = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                model(x)
                latencies.append((time.perf_counter() - start) * 1000)
        return float(np.percentile(latencies, 95))

    def run_onnx_export(self, model: nn.Module) -> Dict[str, float | bool]:
        assert self.input_shape is not None
        self.compressed_dir.mkdir(parents=True, exist_ok=True)

        model = copy.deepcopy(model).to("cpu").float().eval()
        onnx_path = self.compressed_dir / "compressed_model.onnx"
        dummy_input = torch.randn((1, *self.input_shape), dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

        torch_p95 = self._benchmark_torch(model, self.input_shape)
        onnx_p95 = self._benchmark_onnx(onnx_path, self.input_shape)

        faster = onnx_p95 < torch_p95
        self.technique_results["onnx"] = {
            "torch_p95_ms": torch_p95,
            "onnx_p95_ms": onnx_p95,
            "onnx_faster": faster,
            "speedup": torch_p95 / max(onnx_p95, 1e-9),
        }
        return self.technique_results["onnx"]

    def select_and_save_best(self) -> Dict[str, float | str | Dict[str, float]]:
        assert self.original_metrics is not None
        original_acc = self.original_metrics["accuracy"]

        valid = [
            c for c in self.candidates if c[2]["accuracy"] >= (original_acc - 1.0)
        ]
        if not valid:
            valid = [self.candidates[0]]

        winner = min(valid, key=lambda c: c[2]["model_size_mb"])
        winner_name, winner_model, winner_metrics, winner_details = winner

        self.compressed_dir.mkdir(parents=True, exist_ok=True)
        compressed_pt = self.compressed_dir / "compressed_model.pt"
        torch.save(winner_model, compressed_pt)

        onnx_result = self.run_onnx_export(winner_model)

        comparison = compare_models(self.original_metrics, winner_metrics)
        results = {
            "winner": winner_name,
            "input_shape": list(self.input_shape),
            "original": self.original_metrics,
            "compressed": winner_metrics,
            "comparison": comparison,
            "techniques": self.technique_results,
            "winner_details": winner_details,
            "artifacts": {
                "compressed_pt": str(compressed_pt),
                "compressed_onnx": str(self.compressed_dir / "compressed_model.onnx"),
            },
            "onnx_kept": bool(onnx_result["onnx_faster"]),
        }

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
