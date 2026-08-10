#!/usr/bin/env python3

import torch
from unsloth import FastVisionModel

# ============================================================
# CONFIGURATION
# ============================================================

# Base model used during training (prepare_qwen_chat_dataset.py / qwen_sft.py).
BASE_MODEL = "/scratch/proy/models/Qwen3.5-4B"

# Saved LoRA checkpoint. This is `output_dir` / `new_model` from qwen_sft.py.
# Point it at either the final save or a specific checkpoint-XX folder.
VARIANT = "4b"
LORA_MODEL = f"/scratch/atul_prakash/fkg/models/unsloth_Qwen3.5_{VARIANT}_FOOD_FT"

# Must be >= the max_seq_length used in training (we trained at ~8192).
MAX_SEQ_LENGTH = 8192

# How many tokens to generate. Recipes are long, so give it room.
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

# Loading the LoRA folder directly makes Unsloth pull the base weights from the
# adapter config and apply the adapter on top.
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

FastVisionModel.for_inference(model)

text_tokenizer = tokenizer.tokenizer
text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "right"

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

    inputs = text_tokenizer(
        prompt,
        return_tensors="pt",
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
    [
        "Rice",
        "Mutton",
    ],
]

for i, case in enumerate(test_cases, 1):
    pred = predict(case)

    print("=" * 80)
    print(f"Case {i}")
    print("-" * 80)
    print("Ingredients:")
    print("\n".join(f"  - {x}" for x in case) if isinstance(case, list) else case)
    print()
    print("Predicted recipe:")
    print(pred)

print("=" * 80)
