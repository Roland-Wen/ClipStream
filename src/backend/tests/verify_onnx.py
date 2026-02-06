import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

def verify_conversion():
    model_id = "openai/clip-vit-base-patch32"
    onnx_path = "onnx/text_model.onnx"
    test_text = "a professional baseball pitcher throwing a fastball"

    print("1. PyTorch Inference...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    pt_model = CLIPTextModelWithProjection.from_pretrained(model_id)
    pt_model.eval()
    
    # --- FIX START: FORCE PADDING TO 77 ---
    # We add padding="max_length" and max_length=77 to match the ONNX export
    inputs = tokenizer(
        [test_text], 
        padding="max_length",  # Pad to exactly max_length
        max_length=77,         # Standard CLIP context length
        truncation=True,       # Truncate if too long
        return_tensors="pt"
    )
    # --- FIX END ---

    with torch.no_grad():
        pt_out = pt_model(**inputs).text_embeds.numpy()

    print("2. ONNX Inference...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.int64)
    }
    
    onnx_out = session.run(["text_embeds"], onnx_inputs)[0]
    
    print(f"PyTorch Shape: {pt_out.shape}")
    print(f"ONNX Shape:    {onnx_out.shape}")

    sim = cosine_similarity(pt_out, onnx_out)[0][0]
    print(f"Cosine Similarity: {sim:.5f}")
    
    if sim > 0.999:
        print("✅ SUCCESS: Models match perfectly.")
    else:
        print("❌ FAILURE: Models diverge.")

if __name__ == "__main__":
    verify_conversion()