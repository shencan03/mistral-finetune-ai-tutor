import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_DIR = "output/fine-tuned-mistral"
OFFLOAD_DIR = "offload"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,  # already includes load_in_4bit=True
    offload_folder=OFFLOAD_DIR
)

prompt = "Soru: Kuvvet nedir?\nCevap:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

print("\n🧠 Model response:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
