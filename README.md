# NLLB-Tamil-Parameter-Efficient-Fine-Tuning-PEFT-for-High-Fidelity-Translation
# NLLB-Tamil: Fine-Tuning for English-to-Tamil Translation

This project focuses on enhancing Meta's **NLLB-200 (No Language Left Behind)** model specifically for English-to-Tamil translation. By utilizing **LoRA (Low-Rank Adaptation)**, I achieved high-quality translation improvements while training less than 0.2% of the total model parameters.



##  Project Overview
Translation into Dravidian languages like Tamil is challenging due to their **agglutinative** nature (where suffixes are layered onto word roots). This project fine-tunes the 600M parameter distilled NLLB model to better handle these linguistic nuances using the **Samanantar** dataset.

## Key Features
- **Parameter-Efficient Fine-Tuning (PEFT):** Uses LoRA to train only 1.1M parameters out of 616M, allowing for training on consumer-grade hardware.
- **Advanced Evaluation:** Implements **chrF (Character n-gram F-score)** alongside BLEU, which is more accurate for morphologically rich languages like Tamil.
- **Source/Target Precision:** Configured with specific NLLB language tokens (`eng_Latn` and `tam_Taml`) to ensure proper script generation.
- **Optimized Inference:** Includes a beam-search powered inference script for fluent, context-aware translations.

## Tech Stack
- **Foundation Model:** `facebook/nllb-200-distilled-600M`
- **Libraries:** Hugging Face `transformers`, `peft`, `datasets`, `evaluate`
- **Optimization:** LoRA (Low-Rank Adaptation)
- **Dataset:** `ai4bharat/samanantar` (The largest parallel corpus for Indian languages)



## Implementation Details
1. **Target Modules:** Injected LoRA adapters into the `q_proj` and `v_proj` (Attention) layers.
2. **Preprocessing:** Cleaned and tokenized the Samanantar dataset, mapping `src` and `tgt` columns to the model's vocabulary.
3. **Training Strategy:** 3 Epochs with a 2e-4 learning rate and FP16 (if GPU available) or CPU-optimized training.
4. **Metric Scoring:** Tracked model convergence using loss curves and character-level similarity scores.
5. 
## Performance Metrics
- **BLEU Score:** Measures word-level overlap.
- **chrF Score:** Measures character-level similarity (Preferred for Tamil).

## Installation & Use
1. **Install Dependencies:**
   ```bash
   pip install transformers datasets peft evaluate sacrebleu accelerate
