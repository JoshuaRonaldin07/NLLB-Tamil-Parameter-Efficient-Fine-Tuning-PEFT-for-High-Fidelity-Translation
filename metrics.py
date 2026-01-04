import evaluate
import numpy as np

# Load the metrics
metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Decode predictions and labels into strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them (standard NLP practice)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some basic post-processing to clean up whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # Calculate BLEU (Standard)
    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Calculate chrF (Crucial for Tamil)
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"]
    }

# --- Link this to your Trainer ---
# In your main script, update the trainer initialization:
# trainer = Seq2SeqTrainer(
#     ...
#     compute_metrics=compute_metrics,
#     ...
# )
