"""Extract representative keyframes from each video.

Frames are sampled from the video, embedded with an ImageNet ResNet-50 (no
classification head) and clustered with K-Means; the frame closest to each
cluster centre is kept. Frames are saved as
    <output_dir>/<video_id>/keyframe_<i>.jpg
which is the layout expected by the inference scripts (--frames-dir).

Usage:
    python -m preprocessing.extract_keyframes --data-csv data/benchmark.csv \
        --video-dir data/videos --output-dir data/keyframes \
        --num-keyframes <K> --sample-every-sec <S> --resize-width <W>
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import models, transforms
from tqdm import tqdm

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_resnet_embedder(device):
    """ResNet-50 used as a frame feature extractor."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model.eval().to(device)


def embed(model, frame_bgr, device):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        return model(PREPROCESS(pil).unsqueeze(0).to(device)).cpu().numpy().flatten()


def extract_keyframes(video_path, save_dir, model, device, k, sample_every_sec, resize_width, seed):
    cap = cv2.VideoCapture(video_path)
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) * sample_every_sec))

    frames, embeddings = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
            embeddings.append(embed(model, frame, device))
        idx += 1
    cap.release()

    if not frames:
        return 0

    if len(frames) <= k:
        selected = list(range(len(frames)))
    else:
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(X)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        selected = sorted(set(closest.tolist()))  # keep temporal order

    os.makedirs(save_dir, exist_ok=True)
    for i, j in enumerate(selected, start=1):
        frame = frames[j]
        if resize_width:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (resize_width, int(h * resize_width / w)))
        cv2.imwrite(os.path.join(save_dir, f"keyframe_{i}.jpg"), frame)
    return len(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-csv", required=True)
    p.add_argument("--id-col", default="id")
    p.add_argument("--video-dir", required=True, help="Folder containing <video_id>.mp4 files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-keyframes", type=int, required=True, help="Number of K-Means clusters / keyframes per video.")
    p.add_argument("--sample-every-sec", type=float, required=True, help="Sampling interval (seconds) before clustering.")
    p.add_argument("--resize-width", type=int, help="Resize saved frames to this width (aspect ratio kept).")
    p.add_argument("--seed", type=int, help="Random seed for K-Means.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_resnet_embedder(device)
    video_ids = pd.read_csv(args.data_csv)[args.id_col].astype(str).unique()

    for vid in tqdm(video_ids, desc="Extracting keyframes"):
        save_dir = os.path.join(args.output_dir, vid)
        if os.path.isdir(save_dir) and os.listdir(save_dir) and not args.overwrite:
            continue
        video_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: video not found: {video_path}")
            continue
        extract_keyframes(video_path, save_dir, model, device, args.num_keyframes,
                          args.sample_every_sec, args.resize_width, args.seed)


if __name__ == "__main__":
    main()
