"""LLaMA 4 Maverick (GGUF via llama.cpp `llama-server`) inference for IndicVMTBench.

Start the server first with `models/llama4/llama_server.sh` (it loads the vision
projector, so frame-based strategies receive the images), then run
(from the repository root):
    python -m models.llama4.run_inference --data-csv data/benchmark.csv \
        --strategy zero_shot --gen-config configs/generation.yaml
"""

import base64

import requests

from common.io_utils import base_parser, load_gen_config, run


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/v1/chat/completions",
                        help="OpenAI-compatible chat endpoint of llama-server.")
    args = parser.parse_args()

    gen_kwargs = load_gen_config(args.gen_config, "llama4")

    def translate(system, user, image_paths):
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"}}
            for p in image_paths
        ]
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content if image_paths else user})

        response = requests.post(args.server_url, json={"messages": messages, "stream": False, **gen_kwargs})
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    run(args, "llama4", translate)


if __name__ == "__main__":
    main()
