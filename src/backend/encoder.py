import logging
import time
import os
import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer
from typing import List

# Conditional Import: Only import torch if we actually need it
try:
    import torch
    from transformers import CLIPModel
except ImportError:
    torch = None
    CLIPModel = None

logger = logging.getLogger(__name__)

class CLIPTextEncoder:
    _instance = None

    def __init__(self, model_name: str, backend: str = "onnx"):
        self.backend = backend
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        logger.info(f"🚀 Initializing CLIP Encoder | Backend: {self.backend.upper()}")

        # --- BACKEND 1: PYTORCH (Baseline) ---
        if self.backend == "torch":
            if torch is None:
                raise ImportError("Backend 'torch' selected but torch is not installed.")
            self.device = "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

        # --- BACKEND 2: ONNX (Optimized) ---
        elif self.backend == "onnx":
            model_path = os.path.join("onnx", "text_model.onnx")
            self._load_onnx(model_path)

        # --- BACKEND 3: ONNX QUANTIZED (Future W6D4) ---
        elif self.backend == "onnx_quant":
            # We will create this file in W6D4
            model_path = os.path.join("onnx", "text_model_quant.onnx")
            self._load_onnx(model_path)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_onnx(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found at {path}")
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    @classmethod
    def get_instance(cls, model_name: str, backend: str = "onnx"):
        # We assume the backend doesn't change during runtime for the singleton
        if cls._instance is None:
            cls._instance = cls(model_name, backend)
        return cls._instance

    def encode_text(self, text: str) -> List[float]:
        # --- PATH A: PYTORCH ---
        if self.backend == "torch":
            inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features /= features.norm(p=2, dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()

        # --- PATH B: ONNX (Standard & Quantized share this logic) ---
        else:
            inputs = self.tokenizer(
                [text], 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="np"
            )
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            outputs = self.session.run(["text_embeds"], onnx_inputs)
            embeddings = outputs[0]
            
            # Manual Normalization
            norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            normalized = embeddings / (norm + 1e-12)
            
            return normalized[0].tolist()

    def warm_up(self):
        logger.info(f"⚡ Warming up ({self.backend})...")
        start = time.time()
        self.encode_text("warmup query")
        logger.info(f"🔥 Warm-up complete in {round((time.time() - start) * 1000, 2)}ms")