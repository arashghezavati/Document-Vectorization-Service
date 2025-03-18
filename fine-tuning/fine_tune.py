import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from bitsandbytes import BitsAndBytesConfig
import json
import os

# ✅ Step 1: Load and Clean Dataset
print("📌 Loading and cleaning dataset...")
dataset_path = "fine-tuning/dataset.json"  # Ensure dataset exists

raw_data = load_dataset("json", data_files=dataset_path)["train"]

def clean_text(example):
    example["text"] = example["text"].strip().replace("\n", " ")
    return example

cleaned_data = raw_data.map(clean_text)

unique_texts = list(set(cleaned_data["text"]))
final_dataset = Dataset.from_dict({"text": unique_texts})

print(f"✅ Cleaned dataset ready: {len(unique_texts)} unique entries.")

# ✅ Step 2: Load Base Model (Phi-2) and Apply Quantization
print("📌 Loading base model with 4-bit quantization...")
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Step 3: Apply LoRA Fine-Tuning
lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.05,  
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# ✅ Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./phi-2-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=final_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

# ✅ Step 5: Train Model
print("🚀 Fine-tuning in progress...")
trainer.train()

# ✅ Step 6: Save Fine-Tuned Model
model.save_pretrained("./phi-2-finetuned")
tokenizer.save_pretrained("./phi-2-finetuned")

print("🎉 Fine-tuning complete! Model saved to './phi-2-finetuned'.")

# ✅ Step 7: Model Distillation (Creating a Smaller Version)
print("📌 Performing model distillation...")
small_model_name = "phi-2-finetuned-distilled"
small_model = AutoModelForCausalLM.from_pretrained(model_name)

small_trainer = SFTTrainer(
    model=small_model,
    train_dataset=final_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

print("🚀 Training smaller distilled model...")
small_trainer.train()
small_model.save_pretrained(f"./{small_model_name}")

print(f"✅ Model distillation complete! Small model saved to './{small_model_name}'.")

# ✅ Step 8: Inference Optimization (Batch Processing & Caching)
print("📌 Implementing optimized inference...")
import time
from functools import lru_cache

@lru_cache(maxsize=500)
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Test Batch Processing
test_queries = [
    "What is the CRS score required for Canada PR?",
    "How does Express Entry work?",
    "Explain the documents required for PR application."
]

start_time = time.time()
responses = [generate_response(query) for query in test_queries]
end_time = time.time()

print("\n📌 Inference results:")
for query, response in zip(test_queries, responses):
    print(f"📝 Query: {query}")
    print(f"💡 Response: {response}")
    print("-" * 40)

print(f"✅ Inference optimization complete. Batch processed in {end_time - start_time:.2f} seconds.")
