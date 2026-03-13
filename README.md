<div align="center">

# 🗜️ Model Compressor

**Upload any PyTorch model → compress to the smallest possible size → download the optimized artifact**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX](https://img.shields.io/badge/ONNX-1.15-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📸 Screenshot

![Model Compressor UI](assets/screenshot.png)

> *Upload a `.pt` / `.pth` model, watch the real-time compression pipeline, and download the best compressed artifact — all in one page.*

---

## 📌 Overview

**Model Compressor** is a self-contained web application that automatically applies four industry-standard compression techniques to any PyTorch model, benchmarks them, and returns the smallest model that preserves the original accuracy (within 1 %).

| What it does | Why it matters |
|---|---|
| Reduces model file size | Cheaper storage & faster transfer |
| Cuts inference latency | Real-time & edge deployment |
| Exports to ONNX | Cross-platform runtime support |
| Streams live progress | No guessing if something is running |

---

## ✨ Features

- 🔼 **Drag-and-drop upload** — `.pt` / `.pth` files up to 500 MB
- ⚡ **4-technique pipeline** — Quantization → Pruning → Distillation → ONNX Export
- 📡 **Server-Sent Events (SSE)** — real-time step-by-step progress in the browser
- 📊 **Side-by-side metrics** — size, latency (p50 / p95), throughput, accuracy
- 📥 **One-click download** — compressed `.pt` and `.onnx` artifacts
- 🤖 **Architecture-aware** — gracefully skips incompatible techniques (e.g. distillation on YOLO)
- 🔒 **Fully local** — no cloud, no telemetry, everything runs on your machine

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **ML Framework** | [PyTorch](https://pytorch.org) | 2.1.0 |
| **Model Export** | [ONNX](https://onnx.ai) + [ONNX Runtime](https://onnxruntime.ai) | 1.15.0 / 1.16.3 |
| **Object Detection** | [Ultralytics (YOLOv8)](https://ultralytics.com) | 8.4.21 |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com) | 0.104.1 |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org) | 0.24.0 |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev) | 2.5.2 |
| **Numerical** | [NumPy](https://numpy.org) | 1.26.2 |
| **Frontend** | Vanilla HTML / CSS / JS | — |
| **Streaming** | Server-Sent Events (SSE) | — |
| **Runtime** | Python | 3.11 |

---

## 📁 Project Structure

```
model-compression/
│
├── assets/                        # Documentation images
│   └── screenshot.png
│
├── src/                           # Application source code
│   ├── __init__.py
│   ├── compress.py                # CompressionEngine — 4-technique pipeline
│   ├── evaluate.py                # evaluate_model() + compare_models()
│   │
│   └── api/
│       ├── __init__.py
│       ├── server.py              # FastAPI routes + SSE streaming
│       └── static/
│           └── index.html         # Single-page dark-theme UI
│
├── uploads/                       # Temporary storage for uploaded model
│   └── original_model.pt
│
├── compressed/                    # Output artifacts
│   ├── compressed_model.pt
│   └── compressed_model.onnx
│
├── config/
│   └── results.json               # Full metrics written after each run
│
├── requirements.txt               # Pinned dependency list
└── run.py                         # Entry point — starts server + opens browser
```

---

## 🔄 Compression Workflow

```
┌────────────────────────────────────────────────────────┐
│                     User uploads .pt/.pth              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  load_original_model()                                 │
│  • torch.load() with ultralytics fallback              │
│  • Cast to CPU float32                                 │
│  • Detect input shape                                  │
│  • Baseline accuracy + latency                         │
└──────┬─────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────────────┐
       │                                                              │
       ▼                                                              │
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Dynamic   │   │ 2. Pruning   │   │ 3. Knowledge │   │ 4. ONNX      │
│ Quantization │   │ (L1 Unstr.)  │   │ Distillation │   │ Export       │
│              │   │              │   │              │   │              │
│ INT8 via     │   │ 10% → 90%    │   │ Student MLP  │   │ Opset 17     │
│ torch.quant  │   │ sparsity     │   │ trained from │   │ + ORT bench  │
│ ization API  │   │ fast search  │   │ teacher soft │   │ (p50 / p95)  │
│              │   │ (200 samples)│   │ labels       │   │              │
│              │   │              │   │ Skip if YOLO │   │              │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │  select_and_save_best()             │
              │  Pick smallest model where          │
              │  accuracy ≥ original − 1 %          │
              │  → compressed_model.pt              │
              │  → compressed_model.onnx            │
              │  → config/results.json              │
              └─────────────────────────────────────┘
```

### Compression Techniques Explained

| # | Technique | Method | Best for |
|---|---|---|---|
| 1 | **Dynamic Quantization** | Converts weights to INT8 at runtime | Linear/LSTM layers |
| 2 | **Structured Pruning** | L1 unstructured pruning, searches 10 %–90 % sparsity | CNN and Transformer models |
| 3 | **Knowledge Distillation** | Trains compact student MLP from teacher logits | Classic classifiers with linear heads |
| 4 | **ONNX Export** | Exports to ONNX opset 17, benchmarks vs ORT | Cross-platform & latency reduction |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11** (other versions untested)
- Windows / macOS / Linux

### Installation

```bash
# 1 — Clone the repository
git clone https://github.com/your-username/model-compression.git
cd model-compression

# 2 — Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3 — Install dependencies
pip install -r requirements.txt

# 4 — (Optional) YOLO model support
pip install ultralytics==8.4.21 --no-deps
```

### Run

```bash
python run.py
```

The browser opens automatically at **http://localhost:8000**.

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page UI |
| `POST` | `/api/upload` | Upload a `.pt` / `.pth` model (multipart/form-data) |
| `GET` | `/api/compress` | Start compression — SSE stream of progress events |
| `POST` | `/api/compress` | Same as above (browser EventSource uses GET) |
| `GET` | `/api/results` | Return full `results.json` |
| `GET` | `/api/download/compressed` | Download `compressed_model.pt` |
| `GET` | `/api/download/onnx` | Download `compressed_model.onnx` |
| `GET` | `/api/health` | Health check — returns upload/compressed status flags |

### SSE Event Schema

```json
// Step progress
{ "step": "quantization", "status": "running", "progress": 5 }
{ "step": "quantization", "status": "done",    "progress": 25 }

// Completion
{
  "status": "done",
  "results": {
    "winner": "quantization",
    "original":   { "accuracy": 0.91, "model_size_mb": 6.25, "latency_p95_ms": 117 },
    "compressed": { "accuracy": 0.90, "model_size_mb": 1.82, "latency_p95_ms":  14 },
    "comparison": { "size_reduction_percent": 70.9, "speedup": 8.75 }
  }
}
```

---

## 📊 Example Results — YOLOv8n

| Metric | Original | Compressed (ONNX) |
|---|---|---|
| **File size** | 6.25 MB | 6.21 MB |
| **Inference latency (p95)** | 117 ms | 13.4 ms |
| **Speedup** | — | **8.75×** |
| **Distillation** | — | Skipped (YOLO architecture) |

> ONNX Runtime consistently delivers 8–9× faster inference compared to native PyTorch on CPU for YOLO-style models.

---

## ⚙️ Configuration

After each compression run, `config/results.json` is written with the complete report:

```json
{
  "winner": "onnx",
  "input_shape": [1, 3, 640, 640],
  "original": { ... },
  "compressed": { ... },
  "comparison": {
    "size_reduction_percent": 0.6,
    "accuracy_difference": 0.0,
    "speedup": 8.75,
    "compression_ratio": 1.006
  },
  "techniques": {
    "quantization": { ... },
    "pruning":       { ... },
    "distillation":  { "skipped": 1.0 },
    "onnx":          { "onnx_faster": true, "speedup": 8.748 }
  }
}
```

---

## 🧩 Supported Model Formats

| Format | Supported | Notes |
|---|---|---|
| PyTorch `.pt` full model | ✅ | Standard `torch.save(model)` |
| PyTorch `.pth` state dict | ✅ | Auto-detected and handled |
| Ultralytics YOLOv8 `.pt` | ✅ | Requires `ultralytics` installed |
| ONNX `.onnx` input | ❌ | Upload PyTorch checkpoint only |

---

## 🔒 Security Notes

- All processing is **100 % local** — no model data is sent to external servers
- Uploaded files are validated server-side (size limit: 500 MB, extension check)
- The server binds to `127.0.0.1` by default (localhost only)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ using PyTorch · FastAPI · ONNX Runtime

</div>

