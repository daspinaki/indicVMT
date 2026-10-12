"""Dataset and keyframe loading shared by all model runners."""

import os

import pandas as pd
from PIL import Image

# language key -> (display name used in prompts, ground-truth column code)
LANGUAGES = {
    "hindi":     ("Hindi",     "hi"),
    "bengali":   ("Bengali",   "bn"),
    "tamil":     ("Tamil",     "ta"),
    "telugu":    ("Telugu",    "te"),
    "marathi":   ("Marathi",   "mr"),
    "odia":      ("Odia",      "or"),
    "assamese":  ("Assamese",  "as"),
    "punjabi":   ("Punjabi",   "pa"),
    "kannada":   ("Kannada",   "kn"),
    "gujarati":  ("Gujarati",  "gu"),
    "malayalam": ("Malayalam", "ml"),
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_dataset(csv_path, id_col, src_col, desc_col=None):
    """Load the benchmark CSV and return rows as dicts with keys id, english, description."""
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        desc = None
        if desc_col and desc_col in df.columns and pd.notna(r[desc_col]):
            desc = str(r[desc_col])
        rows.append({"id": str(r[id_col]), "english": str(r[src_col]), "description": desc})
    return rows


def frame_paths(frames_dir, video_id, num_frames):
    """Sorted keyframe paths for one video (produced by preprocessing/extract_keyframes.py)."""
    video_dir = os.path.join(frames_dir, video_id)
    if not os.path.isdir(video_dir):
        return []
    files = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(IMAGE_EXTS))
    return [os.path.join(video_dir, f) for f in files[:num_frames]]


def load_frames(paths):
    return [Image.open(p).convert("RGB") for p in paths]
