"""GPT (OpenAI API) inference for IndicVMTBench.

Usage (from the repository root):
    export OPENAI_API_KEY=...
    python -m models.gpt.run_inference --data-csv data/benchmark.csv \
        --strategy zero_shot_tdv --gen-config configs/generation.yaml
"""

import base64
import os

from openai import OpenAI

from common.io_utils import base_parser, load_gen_config, run


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--model-id", default="gpt-4o-mini")
    parser.add_argument("--image-detail", help="OpenAI image detail level (e.g. low / high / auto).")
    args = parser.parse_args()

    gen_kwargs = load_gen_config(args.gen_config, "gpt")
    client = None if args.dry_run else OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def translate(system, user, image_paths):
        content = [{"type": "text", "text": user}]
        for p in image_paths:
            image_url = {"url": f"data:image/jpeg;base64,{encode_image(p)}"}
            if args.image_detail:
                image_url["detail"] = args.image_detail
            content.append({"type": "image_url", "image_url": image_url})

        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content if image_paths else user})

        response = client.chat.completions.create(model=args.model_id, messages=messages, **gen_kwargs)
        return response.choices[0].message.content

    run(args, "gpt", translate)


if __name__ == "__main__":
    main()
