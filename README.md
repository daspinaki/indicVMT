# IndicVMTBench — Inference Code

Inference code for evaluating multimodal LLMs on video-guided machine translation
from English into 11 Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Odia,
Assamese, Punjabi, Kannada, Gujarati, Malayalam) under eight prompting strategies.

Supported models: **GPT-4o-mini**, **Gemini**, **Gemma 3**, **LLaMA 4 Maverick**.

## Repository structure

```
common/
  prompts.py              prompt templates for all strategies (paper, Appendix F)
  exemplars.py            hand-written in-context exemplars per target language
  data.py                 dataset / keyframe loading, language list
  io_utils.py             shared CLI, generation config, output writing
preprocessing/
  extract_keyframes.py    ResNet-50 + K-Means keyframe extraction
  generate_descriptions.py  text description of the video (TDV) with MiniCPM-V
models/
  gpt/run_inference.py
  gemini/run_inference.py
  gemma3/run_inference.py
  llama4/run_inference.py
  llama4/llama_server.sh  launches llama.cpp's llama-server (with vision projector)
configs/
  generation.example.yaml generation parameters (left empty; fill in your own)
```

## Setup

```bash
pip install -r requirements.txt
cp configs/generation.example.yaml configs/generation.yaml   # then fill in values
```

All commands below are run from the repository root.

## Data

The benchmark CSV must contain:

| column | content |
|---|---|
| `id` | video / segment id (frames are looked up in `<frames-dir>/<id>/`) |
| `english` | English source sentence |
| `description` | text description of the video (needed for the TDV strategies) |

Column names can be changed with `--id-col`, `--src-col`, `--desc-col`.

## Preprocessing

**1. Keyframes** (used by the frame-based strategies):

```bash
python -m preprocessing.extract_keyframes --data-csv data/benchmark.csv \
    --video-dir data/videos --output-dir data/keyframes \
    --num-keyframes <K> --sample-every-sec <S> [--resize-width <W>] [--seed <SEED>]
```

**2. Video descriptions (TDV)** with MiniCPM-V, using the description prompt in
`common/prompts.py` (`DESCRIPTION_PROMPT`):

```bash
python -m preprocessing.generate_descriptions --data-csv data/benchmark.csv \
    --video-dir data/videos --output-csv data/benchmark_with_desc.csv \
    --max-frames <N> --sample-every-sec <S>
```

## Prompting strategies

| `--strategy` | Paper | Input to the model |
|---|---|---|
| `zero_shot` | 0-shot | source sentence |
| `zero_shot_video` | 0-shot + V | source sentence + keyframes |
| `zero_shot_tdv` | 0-shot + TDV | source sentence + video description |
| `hindi_fshot_tdv` | Hindi F-shot + TDV | + video description + Hindi exemplars (for every target language) |
| `lang_fshot_tdv` | Lang F-shot + TDV | + video description + target-language exemplars |
| `dependent_cascade` | Dependent Cascade | source sentence + keyframes; the model identifies visual information relevant to the sentence and translates |
| `independent_cascade` | Independent Cascade | source sentence + keyframes; the model identifies visual information in the frames and translates |
| `fewshot_only` | F-shot | source sentence + target-language exemplars (no video information) |

The exact prompt texts are in `common/prompts.py`; all models use the same prompts.

## Running inference

Generation parameters (temperature, max tokens, ...) are read from the
`--gen-config` YAML; anything left `null` falls back to the API/library default.
`--languages` restricts the target languages (default: all 11).
Use `--dry-run` to print the built prompts without calling a model.

**GPT**
```bash
export OPENAI_API_KEY=...
python -m models.gpt.run_inference --data-csv data/benchmark_with_desc.csv \
    --strategy zero_shot_video --frames-dir data/keyframes --num-frames <N> \
    --model-id gpt-4o-mini --gen-config configs/generation.yaml
```

**Gemini**
```bash
export GEMINI_API_KEY=...
python -m models.gemini.run_inference --data-csv data/benchmark_with_desc.csv \
    --strategy lang_fshot_tdv --model-id <gemini-model> --gen-config configs/generation.yaml
```

**Gemma 3** (local GPU)
```bash
python -m models.gemma3.run_inference --data-csv data/benchmark_with_desc.csv \
    --strategy dependent_cascade --frames-dir data/keyframes --num-frames <N> \
    --model-id unsloth/gemma-3-27b-it [--load-in-4bit] --gen-config configs/generation.yaml
```

**LLaMA 4 Maverick** (GGUF served by llama.cpp)
```bash
LLAMA_SERVER_BIN=/path/to/llama-server \
MODEL_PATH=/path/to/Llama-4-Maverick-...-00001-of-0000X.gguf \
MMPROJ_PATH=/path/to/mmproj.gguf \
EXTRA_ARGS="<context size, threads, GPU layers, offloading flags>" \
bash models/llama4/llama_server.sh

python -m models.llama4.run_inference --data-csv data/benchmark_with_desc.csv \
    --strategy zero_shot_tdv --gen-config configs/generation.yaml
```

## Output

One CSV per language:

```
outputs/<model>/<strategy>/<language>.csv     columns: id, english, prediction
```

Rows that failed are stored as `ERROR: ...`; re-running the same command
skips completed rows and retries only the failed ones.
