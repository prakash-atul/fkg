refine_recipes.py

  Reads data/ranveerbrar.json (751 recipes) and writes data/ranveerbrar_refined.json, using the base Qwen model via unsloth (matching your base_infer.py
  conventions — /scratch paths, VARIANTS dict, deterministic do_sample=False decoding).

  Per recipe, Qwen returns strict JSON and the record is written as:
  {
    "id": "...", "title": "...",
    "ingredients": [{"section": "For Sugar Syrup", "name": "Sugar", "quantity": "1 cup"}, ...],
    "steps": ["...", "..."],
    "notes": ["...", "..."]
  }
  - quantity/name split with section headers preserved ("For Tempering" becomes a section, not a phantom ingredient).
  - steps: transcript stripped of storytelling/banter into ordered imperative instructions.
  - notes: chef tips / substitutions / serving suggestions.
  - Prompt hard-constrains the model to only use info present — no invented amounts.

  Robustness (all unit-tested on this box):
  - Unicode fractions (½→1/2), en-dashes, zero-width chars normalized before prompting.
  - Brace-depth JSON extractor that ignores braces inside strings and strips ```json fences.
  - Parse failures are saved with _parse_error/_raw instead of crashing.
  - Resumable: clean records are skipped on restart; parse-failures are retried. Atomic writes every 10 recipes.

  Running it (on the cluster GPU)

  python3 refine_recipes.py --limit 3    # smoke test first
  python3 refine_recipes.py              # full 4b run over all 751
  python3 refine_recipes.py 9b           # larger variant for higher quality

  Config knobs at the top: MAX_SEQ_LENGTH=16384 / MAX_NEW_TOKENS=3072 (sized for the longest ~22.5k-char transcript + 107-line ingredient list) and
  SAVE_EVERY.

  A couple of things worth flagging:
  - I used the base model, not the FOOD_FT LoRA — the LoRA is tuned to generate recipes in a "Dish:\nMethod:" format, which fights JSON extraction. If
  you'd rather use the LoRA, it's a one-line change to VARIANTS/load_model.
  - Since I can't run the GPU here, I couldn't validate actual output quality. I'd suggest running --limit 5 on the cluster and eyeballing the
  steps/notes — if the model is chatty or drops the JSON, tightening SYSTEM_PROMPT or bumping to 9b is the usual fix. Want me to add a small --report
  flag that prints before/after for a few recipes to make that review easier?