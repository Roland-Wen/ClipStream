import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

def run_embedding_quantization():
    model_fp32 = "onnx/text_model.onnx"
    model_pre = "onnx/text_model_pre.onnx"
    model_final = "onnx/text_model_optimized.onnx"

    print("🛠️  Pre-processing...")
    quant_pre_process(model_fp32, model_pre)
    
    # 1. Load the optimized graph
    model = onnx.load(model_pre)
    
    # 2. Identify EVERY MatMul/Gemm node to protect them
    # This ensures the "Brain" stays in FP32
    protected_nodes = [n.name for n in model.graph.node if n.op_type in ['MatMul', 'Gemm']]

    print(f"🛡️  Protecting {len(protected_nodes)} logic nodes. Quantizing Dictionary only...")

    # 3. Quantize everything ELSE (primarily the Gather/Embedding nodes)
    quantize_dynamic(
        model_input=model_pre,
        model_output=model_final,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=protected_nodes
    )
    
    size_final = os.path.getsize(model_final) / (1024 * 1024)
    print(f"\nOptimized model saved at {size_final:.2f} MB")

if __name__ == "__main__":
    run_embedding_quantization()