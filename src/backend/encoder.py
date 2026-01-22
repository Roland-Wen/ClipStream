import logging
import torch
import threading
from transformers import CLIPTokenizer, CLIPModel
from typing import List

logger = logging.getLogger(__name__)

class CLIPTextEncoder:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        logger.info(f"Loading CLIP Model: {model_name}")
        self.device = "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("CLIP Model loaded successfully on CPU.")

    @classmethod
    def get_instance(cls, model_name: str):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_name)
        return cls._instance

    def encode_text(self, text: str) -> List[float]:
        with self._lock:
            inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
                text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
                
            return text_features[0].cpu().numpy().tolist()

    def warm_up(self):
        logger.info("Warming up CLIP model...")
        self.encode_text("warm up query")
        logger.info("Warm-up complete.")