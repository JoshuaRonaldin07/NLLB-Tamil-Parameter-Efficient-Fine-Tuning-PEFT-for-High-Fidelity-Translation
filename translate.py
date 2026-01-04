import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

def translate_en_to_ta(text, base_model_path, adapter_path):
    # 1. Load the Tokenizer
    # We use the base NLLB tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # 2. Load the Base Model
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
    
    # 3. Load the LoRA Adapter (your fine-tuned weights)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 4. Prepare the Input
    # NLLB needs to know the source and target languages
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 5. Generate with Beam Search
    # forced_bos_token_id tells the model to output 'tam_Taml' (Tamil)
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.lang_code_to_id["tam_Taml"], 
        max_length=128,
        num_beams=5
    )

    # 6. Decode and Print
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result

# --- Test it out ---
# path_to_base = "facebook/nllb-200-distilled-600M"
# path_to_adapter = "./nllb-tamil-lora" # Where you saved your trainer output
# print(translate_en_to_ta("The students are learning natural language processing.", path_to_base, path_to_adapter))