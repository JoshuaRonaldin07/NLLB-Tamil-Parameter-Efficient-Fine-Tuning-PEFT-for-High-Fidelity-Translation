import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import( 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, TaskType
import aiohttp
import warnings
from transformers import logging as transformers_logging

# This hides the wall of red text so you only see the progress bar
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
# 1. SETUP CONFIGURATION
model_id = "facebook/nllb-200-distilled-600M"
source_lang = "eng_Latn"
target_lang = "tam_Taml"

# 2. LOAD TOKENIZER AND MODEL
# Adding specific language codes here is the key!
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    src_lang="eng_Latn", 
    tgt_lang="tam_Taml"
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# 3. APPLY LoRA (Parameter Efficient Fine-Tuning)
# This allows you to train on a standard GPU/Laptop
peft_config = LoraConfig(
    task_type="SEQ_2_SEQ_LM", 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. LOAD AND PREPROCESS TAMIL DATASET
raw_datasets = load_dataset("ai4bharat/samanantar", "ta")

# Sanity Check: Print the keys to be 100% sure
print(f"Dataset Columns: {raw_datasets['train'].column_names}")
# Expected output: ['idx', 'src', 'tgt', 'data_source']

# Subset for manageable training
subset_size = 50000 
raw_datasets["train"] = raw_datasets["train"].select(range(subset_size))
dataset = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)

def preprocess_function(examples):
    # Ensure these match the columns 'src' and 'tgt' we saw in your last run
    inputs = examples["src"] 
    targets = examples["tgt"]
    
    # This call will now work because 'tgt_lang' was set above!
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# Now the 'Map' will work!
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. DEFINE EVALUATION METRICS (BLEU & CHRF)
metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    return {
        "bleu": metric.compute(predictions=decoded_preds, references=decoded_labels)["score"],
        "chrf": chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    }

# 6. TRAINING ARGUMENTS
training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-tamil-lora-results",
    eval_strategy="epoch",       # <--- RENAME THIS (remove 'uation')
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), 
    logging_steps=10,
)

# 7. INITIALIZE TRAINER
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

# 8. START TRAINING
if __name__ == "__main__":
    trainer.train()
    # Save the adapter
    model.save_pretrained("./final-tamil-adapter")
    print("Training complete! Adapter saved to ./final-tamil-adapter")