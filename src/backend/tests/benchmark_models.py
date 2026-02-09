import time
import os
import psutil
import gc
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

# --- CONFIGURATION ---
MODELS = {
    "PyTorch (Original)": None,
    "ONNX (FP32)": "onnx/text_model.onnx",
    "ONNX (Optimized)": "onnx/text_model_optimized.onnx"
}

QUERY = "a photo of a baseball player throwing a ball"
ITERATIONS = 100
WARMUP = 10

class BenchmarkRunner:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def get_memory_mb(self):
        return self.process.memory_info().rss / (1024 * 1024)

    def benchmark_pytorch(self):
        print("   🔄 Loading PyTorch Model...")
        mem_base = self.get_memory_mb()
        
        # 1. Load Model (Lazy)
        model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        mem_post_load = self.get_memory_mb()
        
        inputs = self.tokenizer(QUERY, padding="max_length", max_length=77, return_tensors="pt")
        
        # 2. Warmup (Triggers Allocations)
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = model(**inputs)
        
        # 3. Measure True "Active" Memory
        mem_active = self.get_memory_mb()
        
        # Latency Test
        latencies = []
        with torch.no_grad():
            for _ in range(ITERATIONS):
                start = time.time()
                _ = model(**inputs)
                latencies.append((time.time() - start) * 1000)

        # Cleanup
        del model
        gc.collect()
        
        # We return the "Active" footprint (Active - Base)
        return np.mean(latencies), mem_active - mem_base, mem_post_load - mem_base

    def benchmark_onnx(self, model_path):
        print(f"   🔄 Loading ONNX Model: {model_path}...")
        
        if not os.path.exists(model_path):
            return 0, 0, 0

        mem_base = self.get_memory_mb()
        
        # 1. Load Session (Greedy)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        mem_post_load = self.get_memory_mb()

        inputs = self.tokenizer(QUERY, padding="max_length", max_length=77, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # 2. Warmup
        for _ in range(WARMUP):
            session.run(["text_embeds"], ort_inputs)

        # 3. Measure Active Memory
        mem_active = self.get_memory_mb()

        # Latency Test
        latencies = []
        for _ in range(ITERATIONS):
            start = time.time()
            session.run(["text_embeds"], ort_inputs)
            latencies.append((time.time() - start) * 1000)

        del session
        gc.collect()

        return np.mean(latencies), mem_active - mem_base, mem_post_load - mem_base

    def run(self):
        print(f"🚀 Starting Benchmark (Runs: {ITERATIONS}, Warmup: {WARMUP})...")
        
        results = {}
        
        # Test PyTorch
        print("\nTesting PyTorch...")
        lat, mem_active, mem_load = self.benchmark_pytorch()
        results["PyTorch"] = {"latency": lat, "active_ram": mem_active, "load_ram": mem_load}
        
        # Test ONNX
        for name, path in MODELS.items():
            if name == "PyTorch (Original)": 
                continue
            print(f"\nTesting {name}...")
            lat, mem_active, mem_load = self.benchmark_onnx(path)
            if lat > 0:
                results[name] = {"latency": lat, "active_ram": mem_active, "load_ram": mem_load}

        self.print_results(results)
        self.generate_memory_charts(results)
        self.generate_overall_charts(results)

    def print_results(self, results):
        print("\n" + "="*75)
        print(f"{'Model':<20} | {'Latency':<10} | {'Loaded RAM':<12} | {'Active RAM':<12} | {'Increase':<10}")
        print("-" * 75)
        for name, data in results.items():
            diff = data['active_ram'] - data['load_ram']
            print(f"{name:<20} | {data['latency']:<7.2f} ms | {data['load_ram']:<9.2f} MB | {data['active_ram']:<9.2f} MB | +{diff:<6.2f} MB")
        print("="*75 + "\n")

    def generate_memory_charts(self, results):
        names = list(results.keys())
        active_rams = [results[n]["active_ram"] for n in names]
        load_rams = [results[n]["load_ram"] for n in names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(names))
        width = 0.35

        # Stacked Bar Chart for Memory
        ax.bar(x, load_rams, width, label='Initial Load (Cold)', color='#95a5a6')
        ax.bar(x, [active - load for active, load in zip(active_rams, load_rams)], width, bottom=load_rams, label='Runtime Expansion (Hot)', color='#e74c3c')

        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Expansion: Lazy (PyTorch) vs Greedy (ONNX)')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()

        output_path = "../../docs/memory_analysis.png"
        plt.savefig(output_path)
        print(f"✅ Detailed memory chart saved to {output_path}")

    def generate_overall_charts(self, results):
        names = list(results.keys())
        latencies = [results[n]["latency"] for n in names]
        memories = [results[n]["active_ram"] for n in names]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar 1: Latency (Blue)
        x = np.arange(len(names))
        width = 0.35
        bars1 = ax1.bar(x - width/2, latencies, width, label='Latency (ms)', color='#3498db')
        
        ax1.set_ylabel('Latency (ms)', color='#2980b9', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#2980b9')
        ax1.set_title('Inference Performance: PyTorch vs ONNX', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)

        # Bar 2: Memory (Orange)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, memories, width, label='Active RAM (MB)', color='#e67e22')
        ax2.set_ylabel('Active RAM Usage (MB)', color='#d35400', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#d35400')
        
        # Add values on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)
            
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}MB', ha='center', va='bottom', fontsize=9)

        # Legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.tight_layout()
        output_path = "../../docs/benchmark_results.png"
        plt.savefig(output_path)
        print(f"✅ Chart saved to {output_path}")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()