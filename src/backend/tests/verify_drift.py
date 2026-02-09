import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity

def check_drift():
    model_fp32 = "onnx/text_model.onnx"
    model_optimized = "onnx/text_model_optimized.onnx"

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                              revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    test_queries = [
        "a cat sitting on a mat",
        "a huge birthday cake with candles",
        "a vampire flying with her friend in the sky"
    ]
    
    print("1. Loading Models...")
    # Load Baseline
    sess_fp32 = ort.InferenceSession(model_fp32, providers=["CPUExecutionProvider"])
    
    # Load Optimized
    sess_optimized = ort.InferenceSession(model_optimized, providers=["CPUExecutionProvider"])

    print("2. Verifying Accuracy...")
    for query in test_queries:
        inputs = tokenizer(query, padding="max_length", max_length=77, return_tensors="np")
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        
        # Run both (Standard Output)
        out_fp32 = sess_fp32.run(["text_embeds"], ort_inputs)[0]
        out_optimized = sess_optimized.run(["text_embeds"], ort_inputs)[0]
        
        similarity = cosine_similarity(out_fp32, out_optimized)[0][0]
        print(f"Query: '{query[:30]}...' | Similarity: {similarity:.5f}")

if __name__ == "__main__":
    check_drift()