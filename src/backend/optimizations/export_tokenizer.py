from transformers import CLIPTokenizerFast
import os

def export_tokenizer():
    model_id = "openai/clip-vit-base-patch32"
    output_path = "tokenizer.json"  # Saving to src/backend/tokenizer.json
    
    print(f"Loading tokenizer from {model_id}...")
    # Important: Use the 'Fast' version to get the compatible JSON format
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    
    print(f"Saving to {output_path}...")
    tokenizer.backend_tokenizer.save(output_path)
    print("✅ Tokenizer exported successfully!")

if __name__ == "__main__":
    export_tokenizer()