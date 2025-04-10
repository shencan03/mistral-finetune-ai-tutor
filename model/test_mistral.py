from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.1"

print("🔁 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("🔁 Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

print("✅ Model and tokenizer loaded successfully!")
print("💻 Model device map:", model.hf_device_map)
