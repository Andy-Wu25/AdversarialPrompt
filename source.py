import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import json
from tqdm import tqdm  # ✅ Adds a progress bar

# ✅ Show GPU details
print("GPU Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

def main():
    # 1. Load the dataset
    ds = load_dataset("moyix/SecurityEval")

    # 2. Set up model and tokenizer
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    # 3. Use 4-bit quantization with float16 for better speed
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16  # ✅ Faster inference
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # ✅ FIX: Set left padding for decoder-only models
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # ✅ Fixes padding issue

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # 4. Use batch processing to speed up
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=1024,  
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # 5. Process prompts in batches
    batch_size = 4  # Adjust based on GPU memory
    prompts = [entry["Prompt"] for entry in ds["test"]]

    # ✅ Add a progress bar using tqdm
    results = []
    with open("llm_outputs.json", "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Prompts"):
            batch_prompts = prompts[i : i + batch_size]

            # ✅ Generate results for batch
            batched_results = generation_pipeline(
                batch_prompts, 
                batch_size=batch_size, 
                max_length=1024,
                pad_token_id=tokenizer.pad_token_id
            )

            # ✅ Print results live
            for j, result in enumerate(batched_results):
                generated_text = result[0]["generated_text"]
                prompt_id = ds["test"][i + j]["ID"]
                print(f"\n🔹 **Prompt ID {prompt_id}**")
                print(f"📝 **Prompt:** {batch_prompts[j]}")
                print(f"💡 **Generated Code:**\n{generated_text}\n" + "-"*80)

                # ✅ Save to file in real-time
                results.append({"ID": prompt_id, "Prompt": batch_prompts[j], "LLM_Output": generated_text})
                json.dump(results, f, ensure_ascii=False, indent=2)
                f.write("\n")

    print("\n✅ Results saved to `llm_outputs.json`")

if __name__ == "__main__":
    main()
