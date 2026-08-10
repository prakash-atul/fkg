#!/usr/bin/env python3
"""
Refine the raw Ranveer Brar dataset into clean, structured recipes.

The scraped `data/ranveerbrar.json` carries two noisy fields per video:
  - `ingredients`: quantity + name mashed together, with "For ..." section
    headers interleaved and Unicode fractions (½, ¼, ⅓) from the description.
  - `method`     : the raw YouTube subtitle transcript -- full of storytelling,
    banter and filler ("While coming back from Pondicherry ...").

This script asks the base Qwen model to turn each recipe into strict JSON:
    {
      "ingredients": [{"section": ..., "name": ..., "quantity": ...}, ...],
      "steps":       ["...", "..."],          # clean, ordered, no chit-chat
      "notes":       ["...", "..."]           # chef tips / substitutions
    }
and writes the result to `data/ranveerbrar_refined.json`, preserving id/title.

The run is resumable: existing refined records are loaded on startup and their
ids are skipped, so a long cluster job can be restarted without redoing work.

Usage:
    python3 refine_recipes.py                 # 4b base (default)
    python3 refine_recipes.py 9b              # a different base variant
    python3 refine_recipes.py --limit 5       # smoke-test on the first 5
"""

import argparse
import json
import os
import re
import unicodedata

# torch / unsloth are imported lazily inside load_model() and refine_one() so the
# pure-Python helpers (normalisation, JSON parsing) stay importable on a machine
# without the GPU stack -- the model itself runs on the cluster.

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = "/scratch/atul_prakash/fkg"
INPUT_JSON = os.path.join(PROJECT_ROOT, "data", "ranveerbrar.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "ranveerbrar_refined.json")

# Base checkpoints (same layout base_infer.py / qwen_infer.py expect). We use
# the *base* instruct model here, not the FOOD_FT LoRA: this is a structured
# extraction task, not recipe generation, so the untuned model follows the
# JSON-output instruction more faithfully.
VARIANTS = {
    "0.8b": "/scratch/proy/models/Qwen3.5-0.8B",
    "2b": "/scratch/proy/models/Qwen3.5-2B",
    "4b": "/scratch/proy/models/Qwen3.5-4B",
    "9b": "/scratch/proy/models/Qwen3.5-9B",
}

# The longest transcript is ~22.5k chars (~7k tokens) and the longest ingredient
# list has 107 lines, so give the prompt plenty of room plus space to generate.
MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 3072

# Flush to disk every N recipes so a crash loses at most this many.
SAVE_EVERY = 10

# ---- Prompts ----
SYSTEM_PROMPT = (
    "You are a meticulous recipe editor. You are given a dish title, a raw "
    "ingredient list, and a messy auto-generated video transcript of a chef "
    "cooking the dish. Extract a clean, structured recipe. Use ONLY information "
    "that is present in the input -- never invent ingredients, quantities or "
    "steps. Reply with a single JSON object and nothing else."
)

USER_PROMPT = """Dish: {title}

Raw ingredients:
{ingredients}

Transcript:
{method}

Return ONE JSON object with exactly these keys:
- "ingredients": a list of objects, each {{"section": string, "name": string, "quantity": string}}.
    * "section" is the heading the ingredient falls under (e.g. "For Sugar Syrup").
      Use "" if the recipe has no sections. A line like "For Tempering" is a
      section header, not an ingredient -- do not emit it as its own object.
    * "name" is the ingredient only (e.g. "Refined Flour"), no amounts.
    * "quantity" is the amount and any prep note (e.g. "1 cup", "2 tsp, chopped").
      Use "" when no amount is given (e.g. "Salt to taste" -> quantity "to taste").
- "steps": an ordered list of short, clear cooking instructions. Strip all
    storytelling, jokes, greetings and filler from the transcript. Keep only the
    actual actions needed to cook the dish, in order, as imperative sentences.
- "notes": a list of useful chef tips, substitutions or serving suggestions
    mentioned in the transcript. Use an empty list [] if there are none.

Output only the JSON object."""


# ============================================================
# TEXT NORMALISATION
# ============================================================

_FRACTIONS = {
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅕": "1/5",
    "⅛": "1/8",
    "⅜": "3/8",
}


def normalize_text(s):
    """ASCII-ify fractions, drop zero-width junk, collapse whitespace runs."""
    if not s:
        return ""
    for uni, ascii_frac in _FRACTIONS.items():
        s = s.replace(uni, ascii_frac)
    # Zero-width space / non-joiner / joiner and the BOM.
    s = re.sub(r"[​‌‍﻿]", "", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-")
    return s


def clean_ingredients(items):
    return [normalize_text(x).strip() for x in items if normalize_text(x).strip()]


# ============================================================
# MODEL OUTPUT PARSING
# ============================================================

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text):
    """Pull the first balanced JSON object out of the model's reply.

    Returns the parsed dict, or None if nothing parseable is found.
    """
    text = _FENCE_RE.sub("", text.strip())

    start = text.find("{")
    if start == -1:
        return None

    # Walk the string tracking brace depth, ignoring braces inside strings.
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def coerce_record(parsed):
    """Validate/normalise the parsed dict into our output shape.

    Returns (record_dict, ok) where ok is False if it was unusable.
    """
    if not isinstance(parsed, dict):
        return {}, False

    ingredients = []
    for it in parsed.get("ingredients", []) or []:
        if isinstance(it, dict):
            ingredients.append(
                {
                    "section": str(it.get("section", "") or "").strip(),
                    "name": str(it.get("name", "") or "").strip(),
                    "quantity": str(it.get("quantity", "") or "").strip(),
                }
            )
        elif isinstance(it, str):
            ingredients.append({"section": "", "name": it.strip(), "quantity": ""})

    steps = [str(s).strip() for s in (parsed.get("steps") or []) if str(s).strip()]
    notes = [str(n).strip() for n in (parsed.get("notes") or []) if str(n).strip()]

    ok = bool(ingredients) and bool(steps)
    return {"ingredients": ingredients, "steps": steps, "notes": notes}, ok


# ============================================================
# LOAD MODEL
# ============================================================


def load_model(base_model):
    from unsloth import FastVisionModel

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)

    text_tokenizer = tokenizer.tokenizer
    text_tokenizer.pad_token = text_tokenizer.eos_token
    text_tokenizer.padding_side = "left"
    return model, text_tokenizer


def refine_one(model, text_tokenizer, recipe):
    import torch

    ingredients_text = "\n".join(clean_ingredients(recipe.get("ingredients", [])))
    method_text = normalize_text(recipe.get("method", ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                title=recipe.get("title", ""),
                ingredients=ingredients_text,
                method=method_text,
            ),
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
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
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

    raw = text_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return raw


# ============================================================
# MAIN
# ============================================================


def load_done(path):
    """Return {id: record} for already-refined recipes, for resumability."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    # Only treat cleanly-parsed records as done; retry the parse failures.
    return {
        r["id"]: r
        for r in existing
        if isinstance(r, dict) and r.get("id") and not r.get("_parse_error")
    }


def save(path, records_by_id):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(records_by_id.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", nargs="?", default="4b", help="model variant key")
    ap.add_argument("--limit", type=int, default=None, help="process only first N")
    ap.add_argument("--input", default=INPUT_JSON)
    ap.add_argument("--output", default=OUTPUT_JSON)
    args = ap.parse_args()

    base_model = VARIANTS.get(args.variant, VARIANTS["4b"])

    with open(args.input, encoding="utf-8") as f:
        recipes = json.load(f)
    if args.limit:
        recipes = recipes[: args.limit]

    done = load_done(args.output)
    print(f"Total recipes    : {len(recipes)}")
    print(f"Already refined  : {len(done)} (will be skipped)")
    print(f"Output           : {args.output}")

    model, text_tokenizer = load_model(base_model)

    n_ok = n_fail = 0
    for idx, recipe in enumerate(recipes, 1):
        rid = recipe.get("id")
        if rid in done:
            continue

        raw = refine_one(model, text_tokenizer, recipe)
        parsed = extract_json(raw)
        record, ok = coerce_record(parsed) if parsed is not None else ({}, False)

        record["id"] = rid
        record["title"] = recipe.get("title", "")
        if not ok:
            record["_parse_error"] = True
            record["_raw"] = raw
            n_fail += 1
            print(f"[{idx}/{len(recipes)}] PARSE FAIL: {recipe.get('title','')[:60]}")
        else:
            n_ok += 1
            print(
                f"[{idx}/{len(recipes)}] ok: {len(record['ingredients'])} ingredients, "
                f"{len(record['steps'])} steps, {len(record['notes'])} notes "
                f"| {recipe.get('title','')[:50]}"
            )

        done[rid] = record
        if idx % SAVE_EVERY == 0:
            save(args.output, done)

    save(args.output, done)
    print(f"\nDone. ok={n_ok} parse_fail={n_fail} total_records={len(done)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
