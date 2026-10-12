"""Gemini (Google Generative AI API) inference for IndicVMTBench.

Usage (from the repository root):
    export GEMINI_API_KEY=...
    python -m models.gemini.run_inference --data-csv data/benchmark.csv \
        --strategy zero_shot_video --frames-dir data/keyframes --num-frames <N> \
        --model-id <gemini-model-name> --gen-config configs/generation.yaml
"""

import os

import google.generativeai as genai

from common.data import load_frames
from common.io_utils import base_parser, load_gen_config, run


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--model-id", required=True, help="Gemini model name.")
    args = parser.parse_args()

    gen_kwargs = load_gen_config(args.gen_config, "gemini")
    if not args.dry_run:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    models = {}

    def get_model(system):
        # the system prompt is bound to the model object in this SDK
        if system not in models:
            models[system] = genai.GenerativeModel(args.model_id, system_instruction=system)
        return models[system]

    def translate(system, user, image_paths):
        parts = load_frames(image_paths) + [user]
        response = get_model(system).generate_content(
            parts, generation_config=genai.types.GenerationConfig(**gen_kwargs)
        )
        return response.text

    run(args, "gemini", translate)


if __name__ == "__main__":
    main()
