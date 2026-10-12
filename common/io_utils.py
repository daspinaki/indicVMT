"""CLI, generation-config and output helpers shared by all model runners."""

import argparse
import os

import pandas as pd
import yaml
from tqdm import tqdm

from .data import LANGUAGES, frame_paths, load_dataset
from .prompts import STRATEGIES, build_prompt, needs_description, needs_frames


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-csv", required=True, help="Benchmark CSV with source sentences (and TDV descriptions).")
    p.add_argument("--id-col", default="id")
    p.add_argument("--src-col", default="english")
    p.add_argument("--desc-col", default="description", help="Column holding the text description of the video (TDV).")
    p.add_argument("--frames-dir", help="Root folder of keyframes, one sub-folder per video id.")
    p.add_argument("--num-frames", type=int, help="Number of keyframes passed to the model (frame-based strategies).")
    p.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=list(LANGUAGES))
    p.add_argument("--gen-config", help="YAML file with generation parameters (see configs/generation.example.yaml).")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--dry-run", action="store_true", help="Print the prompt of the first row per language without calling the model.")
    return p


def load_gen_config(path, section):
    """Return the generation kwargs for one model; unset (null) keys are dropped."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {k: v for k, v in (cfg.get(section) or {}).items() if v is not None}


def validate_args(args):
    if needs_frames(args.strategy) and (not args.frames_dir or not args.num_frames):
        raise SystemExit(f"Strategy '{args.strategy}' needs --frames-dir and --num-frames.")


def run(args, model_name, translate_fn):
    """Iterate over languages and rows, call translate_fn, and save one CSV per language.

    translate_fn(system, user, image_paths) -> str
    """
    validate_args(args)
    rows = load_dataset(args.data_csv, args.id_col, args.src_col, args.desc_col)
    if needs_description(args.strategy) and any(r["description"] is None for r in rows):
        raise SystemExit(f"Strategy '{args.strategy}' needs a non-empty '{args.desc_col}' column for every row.")

    out_dir = os.path.join(args.output_dir, model_name, args.strategy)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    for lang in args.languages:
        target_name, _ = LANGUAGES[lang]
        out_csv = os.path.join(out_dir, f"{lang}.csv")
        done = {}
        if os.path.exists(out_csv):
            prev = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
            done = {r["id"]: r for _, r in prev.iterrows() if not r["prediction"].startswith("ERROR")}

        results = []
        for row in tqdm(rows, desc=f"{model_name} | {args.strategy} | {lang}"):
            if row["id"] in done:
                results.append(dict(done[row["id"]]))
                continue

            system, user = build_prompt(args.strategy, row["english"], lang, target_name, row["description"])
            images = frame_paths(args.frames_dir, row["id"], args.num_frames) if needs_frames(args.strategy) else []

            if args.dry_run:
                print(f"\n----- {row['id']} | {lang} | frames: {images}\n[system] {system}\n[user]\n{user}")
                break

            if needs_frames(args.strategy) and not images:
                prediction = "ERROR: no frames found"
            else:
                try:
                    prediction = translate_fn(system, user, images).strip()
                except Exception as e:  # keep going; failed rows are retried on the next run
                    prediction = f"ERROR: {e}"
            results.append({"id": row["id"], "english": row["english"], "prediction": prediction})

        if args.dry_run:
            continue
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
