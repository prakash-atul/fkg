#!/usr/bin/env python3
"""
Inference for the food-recipe LoRA.

Usage:
    python3 qwen_infer.py                # 4b adapter (default)
    python3 qwen_infer.py 0.8b           # a different variant
    python3 qwen_infer.py 4b --base      # untuned base model, for comparison
"""

import os
import sys

import torch
from unsloth import FastVisionModel

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.expanduser("~/fkg")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "train_data")

# Base checkpoints stay on /scratch/proy — only the adapters moved to ~/fkg.
VARIANTS = {
    "0.8b": "/scratch/proy/models/Qwen3.5-0.8B",
    "2b": "/scratch/proy/models/Qwen3.5-2B",
    "4b": "/scratch/proy/models/Qwen3.5-4B",
    "9b": "/scratch/proy/models/Qwen3.5-9B",
}

args = [a for a in sys.argv[1:] if not a.startswith("-")]
USE_BASE = "--base" in sys.argv

VARIANT = args[0] if args else "4b"
BASE_MODEL = VARIANTS.get(VARIANT, VARIANTS["4b"])

# Saved LoRA checkpoint: `output_dir` / `new_model` from qwen_sft.py.
# Point it at either the final save or a specific checkpoint-XX folder.
LORA_MODEL = os.path.join(PROJECT_ROOT, "models", f"unsloth_Qwen3.5_{VARIANT}_FOOD_FT")

MODEL_TO_LOAD = BASE_MODEL if USE_BASE else LORA_MODEL

# Read back the value chosen from the data by prepare_qwen_chat_dataset.py so
# inference and training agree. Falls back to the training cap.
seq_len_file = os.path.join(DATA_DIR, "max_seq_length.txt")
if os.path.exists(seq_len_file):
    with open(seq_len_file) as f:
        MAX_SEQ_LENGTH = int(f.read().strip())
else:
    MAX_SEQ_LENGTH = 8192

# How many tokens to generate. Recipes are long, but this must leave room
# inside MAX_SEQ_LENGTH after the prompt.
MAX_NEW_TOKENS = 1024

# ---- Prompts: MUST match prepare_qwen_chat_dataset.py exactly ----
SYSTEM_PROMPT = (
    "You are a professional Indian chef. "
    "Given a list of ingredients, suggest a suitable dish and "
    "write clear step-by-step cooking instructions to prepare it."
)

USER_PROMPT = """Given the following ingredients, suggest a dish and explain how to cook it.

Ingredients:
{ingredients}
"""

# ============================================================
# LOAD MODEL
# ============================================================

print(f"Variant        : {VARIANT}")
print(f"Loading        : {MODEL_TO_LOAD}")
print(f"Mode           : {'BASE (untuned)' if USE_BASE else 'LoRA fine-tuned'}")
print(f"max_seq_length : {MAX_SEQ_LENGTH}")

# Loading the LoRA folder directly makes Unsloth pull the base weights from the
# adapter config and apply the adapter on top.
# load_in_4bit=True matches qwen_sft.py, which trained with 4-bit QLoRA.
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_TO_LOAD,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

FastVisionModel.for_inference(model)

# Sanity check: confirm the adapter actually attached. If this prints nothing
# when it should, you are silently running the base model.
if not USE_BASE:
    peft_config = getattr(model, "peft_config", None)
    if peft_config:
        for name, cfg in peft_config.items():
            print(
                f"Adapter '{name}': r={cfg.r}, alpha={cfg.lora_alpha}, "
                f"base={cfg.base_model_name_or_path}"
            )
    else:
        print("WARNING: no peft_config found — the LoRA adapter is NOT attached.")

text_tokenizer = tokenizer.tokenizer
text_tokenizer.pad_token = text_tokenizer.eos_token
# Left padding is what generation wants; right padding misaligns the prompt
# against the generated tokens as soon as you batch more than one case.
text_tokenizer.padding_side = "left"

# ============================================================
# PREDICTION
# ============================================================


def predict(ingredients):
    # Accept either a list of ingredients or a preformatted string, matching
    # how format_example() built the training text.
    if isinstance(ingredients, list):
        ingredients_text = "\n".join(str(i) for i in ingredients)
    else:
        ingredients_text = str(ingredients)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT.format(ingredients=ingredients_text),
        },
    ]

    prompt = text_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # The chat template already emits every special token the model expects;
    # add_special_tokens=False matches how lengths were measured at prep time.
    inputs = text_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=text_tokenizer.eos_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
        )

    prediction = text_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return prediction


# ============================================================
# TEST CASES  (ingredient lists)
# ============================================================

test_cases = [
    ["Refined Flour", "Yogurt"],
    ["Rice", "Mutton"],
]

n_in_format = 0

for i, case in enumerate(test_cases, 1):
    pred = predict(case)

    if pred.startswith("Dish:"):
        n_in_format += 1

    print("=" * 80)
    print(f"Case {i}")
    print("-" * 80)
    print("Ingredients:")
    print("\n".join(f"  - {x}" for x in case) if isinstance(case, list) else case)
    print()
    print("Predicted recipe:")
    print(pred)

print("=" * 80)

# Training targets all begin with "Dish: ...". A low count here means the
# fine-tune has not taken hold, regardless of how good the prose looks.
print(f"Outputs in trained 'Dish:' format: {n_in_format}/{len(test_cases)}")
