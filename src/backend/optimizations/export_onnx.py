import torch
import os
from transformers import CLIPTextModelWithProjection

def export_manual():
    model_id = "openai/clip-vit-base-patch32"
    output_dir = "onnx"
    output_path = os.path.join(output_dir, "text_model.onnx")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"1. Loading Text Encoder from {model_id}...")
    # specifically load the Text-Tower-With-Projection (gives us the 512-dim vector)
    model = CLIPTextModelWithProjection.from_pretrained(model_id)
    model.eval()

    # 2. Create Dummy Inputs (Batch Size 1, Seq Length 77)
    print("2. Creating dummy inputs...")
    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.long)
    dummy_mask = torch.ones((1, 77), dtype=torch.long)

    # 3. Define a Wrapper to clean up outputs
    # Transformers models return complex objects (BaseModelOutput). 
    # We just want the 'text_embeds' tensor.
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model
            
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Return ONLY the projected text embedding (Shape: Batch, 512)
            return outputs.text_embeds

    wrapper = TextEncoderWrapper(model)

    print(f"3. Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_mask),     # Inputs for the trace
        output_path,
        export_params=True,            # Store weights inside the file
        opset_version=14,              # Good standard version
        do_constant_folding=True,      # Optimization
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],  # Naming the output node
        dynamic_axes={                 # Allow variable batch sizes
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embeds": {0: "batch_size"}
        }
    )
    
    print("✅ Export Complete! This model strictly accepts text inputs.")

if __name__ == "__main__":
    export_manual()