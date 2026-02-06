import logging
import time
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import List

logger = logging.getLogger(__name__)

class CLIPTextEncoder:
    _instance = None

    def __init__(self, model_name: str = None):
        # 1. Load the standalone Tokenizer
        tokenizer_path = "tokenizer.json"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_path}. Did you run export_tokenizer.py?")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # CLIP specific: enable padding and truncation
        self.tokenizer.enable_padding(pad_id=0, pad_token="<|endoftext|>", length=77)
        self.tokenizer.enable_truncation(max_length=77)

        # 2. Load the Optimized ONNX Model
        # We target the hybrid quantized model
        model_path = os.path.join("onnx", "text_model_optimized.onnx")
        
        if not os.path.exists(model_path):
            # Fallback to standard ONNX if optimized version is missing
            logger.warning(f"⚠️ Optimized model not found at {model_path}. Checking for standard ONNX...")
            model_path = os.path.join("onnx", "text_model.onnx")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CRITICAL: No ONNX model found at {model_path}")

        logger.info(f"🚀 Loading Inference Engine: {model_path}")
        
        # CPU Execution Provider is default for this project
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        logger.info("✅ Encoder initialized successfully (No Torch/Transformers dependencies).")

    @classmethod
    def get_instance(cls):
        # Singleton pattern
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_text(self, text: str) -> List[float]:
        # 1. Tokenize (Fast Rust Implementation)
        encoded = self.tokenizer.encode(text)
        
        # 2. Prepare Inputs for ONNX
        # Note: tokenizers returns 'ids' and 'attention_mask' directly
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # 3. Inference
        outputs = self.session.run(["text_embeds"], onnx_inputs)
        embeddings = outputs[0]

        # 4. L2 Normalization (Crucial for Cosine Similarity)
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        normalized = embeddings / (norm + 1e-12)

        return normalized[0].tolist()

    def warm_up(self):
        logger.info("⚡ Warming up ONNX engine...")
        start = time.time()
        self.encode_text("warmup query")
        logger.info(f"🔥 Warm-up complete in {round((time.time() - start) * 1000, 2)}ms")