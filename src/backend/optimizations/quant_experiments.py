import os
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from transformers import CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_FP32 = "onnx/text_model.onnx"
MODEL_PRE = "onnx/text_model_pre.onnx"
TEST_QUERY = "a cat sitting on a mat" 

def get_similarity(model_path):
    sess_base = ort.InferenceSession(MODEL_FP32, providers=["CPUExecutionProvider"])
    sess_test = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    inputs = tokenizer(TEST_QUERY, padding="max_length", max_length=77, return_tensors="np")
    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    
    emb_base = sess_base.run(["text_embeds"], ort_inputs)[0]
    emb_test = sess_test.run(["text_embeds"], ort_inputs)[0]
    
    return cosine_similarity(emb_base, emb_test)[0][0]

def run_experiment():
    print("🛠️  Pre-processing...")
    quant_pre_process(MODEL_FP32, MODEL_PRE)
    
    model = onnx.load(MODEL_PRE)
    all_matmuls = [n.name for n in model.graph.node if n.op_type == 'MatMul']
    
    results = []
    # Test keeping 0, 1, 2, 3, 6, or ALL (12) layers in FP32
    for keep_last_n in [0, 1, 2, 3, 6, 8, 10, 12]:
        print(f"\n🧪 Testing: Protecting last {keep_last_n} layers + Projection...")
        
        # 1. Start with the final projection (always sensitive)
        exclude_nodes = [n for n in all_matmuls if "text_projection" in n]
        
        # 2. Add specific encoder layers to exclusion
        start_layer = 12 - keep_last_n
        for i in range(start_layer, 12):
            # Matches your specific naming: layers.X
            layer_nodes = [n for n in all_matmuls if f"layers.{i}/" in n]
            exclude_nodes.extend(layer_nodes)
            
        temp_path = f"onnx/temp_hybrid_{keep_last_n}.onnx"
        
        quantize_dynamic(
            model_input=MODEL_PRE,
            model_output=temp_path,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=exclude_nodes
        )
        
        sim = get_similarity(temp_path)
        size = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"   📊 Similarity: {sim:.5f} | Size: {size:.1f} MB")
        results.append((keep_last_n, size, sim))
        os.remove(temp_path)

    print("\n🏆 --- HYBRID QUANTIZATION RESULTS ---")
    print(f"{'FP32 Layers':<12} | {'Size (MB)':<10} | {'Similarity':<10}")
    for n, s, sim in results:
        print(f"{n:<12} | {s:<10.1f} | {sim:.5f}")

if __name__ == "__main__":
    run_experiment()