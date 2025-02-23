# 1. Install necessary libraries if not already done:
#    pip install datasets transformers accelerate

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json

def main():
    # 2. Load the SecurityEval dataset
    ds = load_dataset("moyix/SecurityEval")
    
    # 3. Specify the Llama model you want to use
    #    Example: meta-llama/Llama-2-7b-chat-hf, or your custom path.
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    # 4. Load the tokenizer and model
    #    device_map="auto" will attempt to place model layers on available GPU(s).
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    # 5. Create a text generation pipeline
    #    You can adjust parameters like max_length, temperature, top_p, etc.
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,    # maximum number of tokens in the generated text
        temperature=0.7,   # controls randomness of outputs
        top_p=0.9,         # nucleus sampling
        do_sample=True
    )
    
    # 6. Iterate over each prompt in the test split, run the LLM, store results
    results = []
    for entry in ds["test"]:
        # Extract the prompt
        prompt_text = entry["Prompt"]
        item_id = entry["ID"]
        
        # Generate response from the model
        output = generation_pipeline(prompt_text)[0]["generated_text"]
        
        # Collect results
        results.append({
            "ID": item_id,
            "Prompt": prompt_text,
            "LLM_Output": output
        })
    
    # 7. Save all results to a JSON file
    out_filename = "llm_outputs.json"
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {out_filename}")

if __name__ == "__main__":
    main()
