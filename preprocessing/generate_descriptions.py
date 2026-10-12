"""Generate the text description of each video (TDV) with MiniCPM-V.

Descriptions are generated once, offline, with the description prompt of the
paper (Fig. 18) and stored in a new `description` column, so that every
translation model receives exactly the same description.

Usage:
    python -m preprocessing.generate_descriptions --data-csv data/benchmark.csv \
        --video-dir data/videos --output-csv data/benchmark_with_desc.csv \
        --max-frames <N> --sample-every-sec <S>
"""

import argparse
import os

import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common.prompts import DESCRIPTION_PROMPT


def uniform_sample(items, n):
    gap = len(items) / n
    return [items[int(i * gap + gap / 2)] for i in range(n)]


def encode_video(video_path, max_frames, sample_every_sec):
    vr = VideoReader(video_path, ctx=cpu(0))
    step = max(1, round(vr.get_avg_fps() * sample_every_sec))
    frame_idx = list(range(0, len(vr), step))
    if len(frame_idx) > max_frames:
        frame_idx = uniform_sample(frame_idx, max_frames)
    return [Image.fromarray(f.astype("uint8")) for f in vr.get_batch(frame_idx).asnumpy()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-csv", required=True)
    p.add_argument("--id-col", default="id")
    p.add_argument("--video-dir", required=True, help="Folder containing <video_id>.mp4 files.")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--model-id", default="openbmb/MiniCPM-V-2_6")
    p.add_argument("--max-frames", type=int, required=True, help="Maximum frames given to MiniCPM-V per video.")
    p.add_argument("--sample-every-sec", type=float, required=True, help="Frame sampling interval in seconds.")
    p.add_argument("--max-slice-nums", type=int, help="MiniCPM-V image slicing parameter.")
    args = p.parse_args()

    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    chat_kwargs = {"use_image_id": False}
    if args.max_slice_nums:
        chat_kwargs["max_slice_nums"] = args.max_slice_nums

    df = pd.read_csv(args.data_csv)
    descriptions = []
    for vid in tqdm(df[args.id_col].astype(str), desc="Describing videos"):
        video_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: video not found: {video_path}")
            descriptions.append(None)
            continue
        frames = encode_video(video_path, args.max_frames, args.sample_every_sec)
        msgs = [{"role": "user", "content": frames + [DESCRIPTION_PROMPT]}]
        descriptions.append(model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **chat_kwargs))

    df["description"] = descriptions
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
