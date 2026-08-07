#!/usr/bin/env python3

import os
from datasets import load_dataset
from unsloth import FastVisionModel

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_JSON = "/scratch/atul_prakash/fkg/data/ranveerbrar.json"
# Local test path (uncomment when running on your machine):
# INPUT_JSON = "ranveerbrar.json"

OUTPUT_DIR = "/scratch/atul_prakash/fkg/data/train_data/"

OUT_TRAIN = os.path.join(OUTPUT_DIR, "train_chat")
OUT_VALID = os.path.join(OUTPUT_DIR, "validation_chat")

VAL_SIZE = 100
SEED = 42

# Percentile of the token-length distribution to cover, and the multiple to
# round the chosen max_seq_length up to. p100 == the longest example, i.e. zero
# truncation, as long as it fits under MAX_SEQ_CAP.
LENGTH_PERCENTILE = 100
ROUND_TO = 256
# Hard ceiling. 8192 comfortably covers the longest recipe in this dataset.
MAX_SEQ_CAP = 8192
# Written here so qwen_sft.py can read back the value chosen from the data.
SEQ_LEN_FILE = os.path.join(OUTPUT_DIR, "max_seq_length.txt")


MODEL_PATH = "/scratch/proy/models/Qwen3.5-4B"

_, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=False,
)

text_tokenizer = tokenizer.tokenizer
text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "right"

SYSTEM_PROMPT = (
    "You are a professional Indian chef. "
    "Given a list of ingredients, suggest a suitable dish and "
    "write clear step-by-step cooking instructions to prepare it."
)

USER_PROMPT = """Given the following ingredients, suggest a dish and explain how to cook it.

Ingredients:
{ingredients}
"""


def format_example(example):

    # Input: the ingredient list (one item per line).
    ingredients = example["ingredients"]
    if isinstance(ingredients, list):
        ingredients_text = "\n".join(str(i) for i in ingredients)
    else:
        ingredients_text = str(ingredients)

    # Output: the dish name (title) followed by the cooking method.
    answer = f"Dish: {example['title']}\n\nMethod:\n{example['method']}"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT.format(ingredients=ingredients_text),
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    text = text_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    return {"text": text}


def preprocess(dataset):
    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting chat prompts",
    )


def choose_max_seq_length(dataset):
    """Tokenize the formatted text and pick a max_seq_length from the data."""
    lengths = [
        len(text_tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in dataset["text"]
    ]
    lengths.sort()

    def pct(p):
        idx = min(len(lengths) - 1, int(p / 100 * len(lengths)))
        return lengths[idx]

    target = pct(LENGTH_PERCENTILE)
    # Round up to the nearest ROUND_TO, then clamp to the ceiling.
    chosen = min(MAX_SEQ_CAP, ((target + ROUND_TO - 1) // ROUND_TO) * ROUND_TO)

    print("\nToken length stats (formatted chat text):")
    print(f"  min / median / max : {lengths[0]} / {pct(50)} / {lengths[-1]}")
    print(f"  p90 / p95 / p99    : {pct(90)} / {pct(95)} / {pct(99)}")
    print(f"  p{LENGTH_PERCENTILE} target        : {target}")
    print(f"  chosen max_seq_len : {chosen} (cap {MAX_SEQ_CAP})")
    n_trunc = sum(1 for x in lengths if x > chosen)
    print(f"  truncated examples : {n_trunc} / {len(lengths)}")

    return chosen


def main():
    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=INPUT_JSON,
        split="train",
    )

    print(f"Total examples: {len(dataset)}")

    if len(dataset) <= VAL_SIZE:
        raise ValueError(
            f"Dataset contains only {len(dataset)} examples. "
            f"Need more than {VAL_SIZE} examples."
        )

    # Random train/validation split
    split = dataset.train_test_split(
        test_size=VAL_SIZE,
        seed=SEED,
        shuffle=True,
    )

    train = split["train"]
    valid = split["test"]

    print(f"Train: {len(train)}")
    print(f"Validation: {len(valid)}")

    print("\nFormatting datasets...")

    train = preprocess(train)
    valid = preprocess(valid)

    max_seq_length = choose_max_seq_length(train)

    print("\nSaving...")

    train.save_to_disk(OUT_TRAIN)
    valid.save_to_disk(OUT_VALID)

    with open(SEQ_LEN_FILE, "w") as f:
        f.write(str(max_seq_length))
    print(f"Wrote max_seq_length={max_seq_length} to {SEQ_LEN_FILE}")

    print("\nDone!")

    print("\nSample:\n")
    print(train[0]["text"])


if __name__ == "__main__":
    main()
