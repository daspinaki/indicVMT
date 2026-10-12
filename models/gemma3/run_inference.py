"""Gemma 3 (local, Unsloth) inference for IndicVMTBench.

Usage (from the repository root):
    python -m models.gemma3.run_inference --data-csv data/benchmark.csv \
        --strategy dependent_cascade --frames-dir data/keyframes --num-frames <N> \
        --gen-config configs/generation.yaml
"""

import torch

from common.data import load_frames
from common.io_utils import base_parser, load_gen_config, run


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--model-id", default="unsloth/gemma-3-27b-it")
    parser.add_argument("--max-seq-length", type=int, help="Context length passed to Unsloth.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    args = parser.parse_args()

    gen_kwargs = load_gen_config(args.gen_config, "gemma3")

    model = processor = None
    if not args.dry_run:
        from unsloth import FastModel  # import unsloth before transformers
        from transformers import AutoProcessor

        load_kwargs = {"model_name": args.model_id, "load_in_4bit": args.load_in_4bit}
        if args.max_seq_length:
            load_kwargs["max_seq_length"] = args.max_seq_length
        model, _ = FastModel.from_pretrained(**load_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_id)

    def translate(system, user, image_paths):
        images = load_frames(image_paths)
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": user}],
        })

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=images or None, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        return processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    run(args, "gemma3", translate)


if __name__ == "__main__":
    main()
