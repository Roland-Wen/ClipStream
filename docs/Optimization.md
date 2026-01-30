# 🚀 ClipStream Optimization Log

## Phase 2, Week 6: Inference Optimization

### **1. Model Export: PyTorch to ONNX (W6D1)**
* **Target Model:** `openai/clip-vit-base-patch32` (Text Encoder Only)
* **Motivation:** The standard PyTorch runtime includes heavy overhead (autograd, gradient tracking) that isn't needed for production inference. ONNX (Open Neural Network Exchange) allows for graph-level optimizations and is significantly faster on CPU-only environments.

#### **The Export Strategy**
We performed a **Manual Export** rather than using automated CLI tools to decouple the Text Tower from the Vision Tower. This reduced the model size and removed unnecessary dependencies on image input tensors.

* **Logic:** Wrapped `CLIPTextModelWithProjection` in a custom `nn.Module`.
* **Input Signature:** `(input_ids [batch, 77], attention_mask [batch, 77])`
* **Output Signature:** `(text_embeds [batch, 512])`
* **Opset Version:** 14

#### **Verification Results**
| Metric | Result |
| :--- | :--- |
| **PyTorch Shape** | `(1, 512)` |
| **ONNX Shape** | `(1, 512)` |
| **Cosine Similarity** | **1.00000** ✅ |

> **Note:** Perfect similarity was achieved by ensuring the tokenizer uses a fixed context length of **77** (`padding='max_length'`) to match the static graph dimensions exported during the trace.

---

### **2. Performance Benchmarking (TBD)**
* **Baseline (PyTorch CPU):** ~120ms / query
* **Optimized (ONNX CPU):** *Pending Week 6, Day 3 tests*
* **Memory Footprint:** 
    * Full CLIP (PT): ~600MB
    * Text-Only (ONNX): ~240MB ( ~60% reduction)

---

### **3. Upcoming Optimizations**
* [ ] **Quantization (W6D4):** Converting weights from `FP32` to `INT8` to further reduce latency.
* [ ] **FastAPI Integration:** Replacing the `CLIPTextEncoder` implementation with an `onnxruntime` session.