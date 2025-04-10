from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "mistralai/Mistral-7B-v0.1"

print("🔁 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 🧠 Sample question (MEB-style)
prompt = "Soru: Kuvvet nedir?\nCevap:"

print("🧠 Generating response...")
output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]

print("\n🎯 Output:\n")
print(output)
