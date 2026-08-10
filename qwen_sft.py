import sys
import os
import warnings

from datasets import load_from_disk
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from transformers import EarlyStoppingCallback

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================
# CONFIGURATION
# =====================================================
VARIANT = "0.8b"
if len(sys.argv) > 1:
    VARIANT = sys.argv[1]

VARIANTS = {
    "0.8b": "/scratch/proy/models/Qwen3.5-0.8B",
    "2b": "/scratch/proy/models/Qwen3.5-2B",
    "4b": "/scratch/proy/models/Qwen3.5-4B",
    "9b": "/scratch/proy/models/Qwen3.5-9B",
}

# ============================================================
# Step-1: File Paths
# ============================================================
# Project now lives under the home directory. Base checkpoints stay on
# /scratch/proy/models (see VARIANTS above).
# DATA_DIR MUST match OUTPUT_DIR in prepare_qwen_chat_dataset.py.
PROJECT_ROOT = os.path.expanduser("~/fkg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train_data")

train_dataset = load_from_disk(os.path.join(DATA_DIR, "train_chat"))
eval_dataset = load_from_disk(os.path.join(DATA_DIR, "validation_chat"))

print(f"Train examples: {len(train_dataset)}")
print(f"Val examples: {len(eval_dataset)}")

print(train_dataset.column_names)
print(train_dataset[0]["text"])


# ============================================================
# Step-2: Sequence length and batch scaling
# ============================================================
# max_seq_length is derived from the dataset in prepare_qwen_chat_dataset.py
# (LENGTH_PERCENTILE of token lengths, rounded up to ROUND_TO, capped at
# MAX_SEQ_CAP). Fall back to 2048 if the file is missing.
seq_len_file = os.path.join(DATA_DIR, "max_seq_length.txt")
if os.path.exists(seq_len_file):
    with open(seq_len_file) as f:
        max_seq_length = int(f.read().strip())
    print(f"Loaded max_seq_length={max_seq_length} from {seq_len_file}")
else:
    max_seq_length = 2048
    print(f"WARNING: {seq_len_file} not found; using max_seq_length={max_seq_length}")

# Auto-scale batch size inversely with sequence length so peak activation memory
# stays roughly constant regardless of the max_seq_length chosen from the data.
# Reference point: batch 32 fit at seq 2048 -> a per-device token budget of 64k.
# gradient_accumulation_steps is then adjusted to hold the effective batch (~64)
# constant, so the learning-rate schedule doesn't change with sequence length.
TOKEN_BUDGET = 32 * 2048  # tokens per device per micro-step
EFFECTIVE_BATCH = 64  # = old batch(32) * old grad_accum(2)

per_device_train_batch_size = max(1, TOKEN_BUDGET // max_seq_length)
gradient_accumulation_steps = max(1, EFFECTIVE_BATCH // per_device_train_batch_size)
effective_batch = per_device_train_batch_size * gradient_accumulation_steps
print(
    f"Auto-scaled: per_device_train_batch_size={per_device_train_batch_size}, "
    f"gradient_accumulation_steps={gradient_accumulation_steps} "
    f"(effective batch ~{effective_batch})"
)

# Evaluate/checkpoint a fixed number of times per epoch rather than on a fixed
# step count. With a small dataset, eval_steps=5 fires many times inside epoch 1
# and early stopping can halt training before the adapter has learned anything.
EVALS_PER_EPOCH = 2
steps_per_epoch = max(1, len(train_dataset) // effective_batch)
eval_steps = max(1, steps_per_epoch // EVALS_PER_EPOCH)
save_steps = eval_steps  # must match eval_steps for load_best_model_at_end
print(
    f"steps_per_epoch~{steps_per_epoch} -> eval_steps={eval_steps}, "
    f"save_steps={save_steps}"
)

base_adapter = VARIANTS.get(VARIANT, VARIANTS["4b"])
output_dir = os.path.join(PROJECT_ROOT, "models", f"unsloth_Qwen3.5_{VARIANT}_FOOD_FT")

new_model = output_dir

# ============================================================
# Step-3: Model Loading
# ============================================================
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=base_adapter,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Extract your inner text tokenizer
text_tokenizer = tokenizer.tokenizer
text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "right"

# ============================================================
# Step-4: Allocate LoRA adapters
# ============================================================
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

model.print_trainable_parameters()

# ============================================================
# Step-5: SFTTrainer Setup
# ============================================================
FastVisionModel.for_training(model)

args = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True,
    eval_accumulation_steps=1,
    optim="paged_adamw_32bit",
    bf16=True,
    max_seq_length=max_seq_length,
    output_dir=output_dir,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=text_tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0,
        )
    ],
)

# Apply prompt masking to calculate loss ONLY on the assistant's response
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ============================================================
# Step-6: Model Training & Saving
# ============================================================
trainer.train()

# Save final adapters locally
model.save_pretrained(new_model)
text_tokenizer.save_pretrained(new_model)
print(f"Saved adapter to {new_model}")
