import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig    

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    "models/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "models/Mistral-7B-v0.1"
)

inputs = tokenizer("Merhaba! Bu bir testtir.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
