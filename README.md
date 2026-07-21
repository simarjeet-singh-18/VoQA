# Audio-Visual QA Benchmark

Benchmarking pipeline for the audio-visual multiple-choice QA dataset
(video + audio, text-overlay question, N-way multiple choice answer).
Built to be extensible: add one model at a time, re-run, and aggregate
into a leaderboard for the paper.

## Project layout

```
avqa_benchmark/
├── dataset.py          # loads the JSON + resolves media paths, sanity checks
├── metrics.py           # accuracy overall + by question_type + by question_relation
├── evaluate.py           # main entrypoint: run one model over the dataset
├── leaderboard.py        # aggregate all results/*.json into one CSV
├── models/
│   ├── base_model.py      # BenchmarkModel interface every model wrapper implements
│   ├── random_model.py    # trivial random baseline (no GPU needed)
│   └── qwen_omni_model.py # Qwen2.5-Omni-7B wrapper (video+audio native)
├── requirements.txt
└── results/               # output JSON per model run (created automatically)
```

## Expected data layout

Point `--base_dir` at the folder containing the sub-directories already
referenced by the JSON's `*_path` fields:

```
<base_dir>/
├── video_audio/<id>.mp4   # video WITH audio (what models are evaluated on)
├── video_muted/<id>.mp4
├── audio_mp3/<id>.mp3
└── metadata/<id>.json
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Sanity-check the dataset loads and media files resolve correctly
python dataset.py --json_path data.json --base_dir /path/to/dataset_root

# 2. Pipeline smoke test with zero dependencies (random baseline)
python evaluate.py --json_path data.json --base_dir /path/to/dataset_root \
    --model random --limit 20

# 3. Real benchmark run (needs a CUDA GPU; see models/qwen_omni_model.py header)
python evaluate.py --json_path data.json --base_dir /path/to/dataset_root \
    --model qwen2.5-omni --out_dir results

# 4. After running a few models, build the comparison table
python leaderboard.py --results_dir results --out_csv leaderboard.csv
```

Each `evaluate.py` run writes one JSON to `results/`, containing:
- overall accuracy
- accuracy broken down by `question_type` (e.g. "Happening", "Come From")
- accuracy broken down by `question_relation` (e.g. "View", "Both")
- the full per-sample predictions + raw model text, for error analysis

## Adding a new model

1. Create `models/your_model.py`, subclass `BenchmarkModel` (see
   `models/base_model.py`), implement `predict(self, sample) ->
   PredictionResult`.
2. Register it in `get_model()` inside `evaluate.py`.
3. Run `python evaluate.py --model your_model_key ...`.

Every model receives the same `AVQASample` object and the same
`build_prompt()` helper (override it per-model only if a model needs a
different prompt format, e.g. GPT-4o's message schema vs. Qwen's chat
template) — this keeps prompts comparable across models for the paper.

For API-based models (GPT-4o, Gemini, Claude, etc.) the wrapper just
needs to base64-encode the video/audio and call the provider's API
instead of loading local weights; the rest of the pipeline (metrics,
leaderboard) is unchanged.

## Notes / caveats

- `models/qwen_omni_model.py` downloads weights from huggingface.co and
  requires a CUDA GPU (7B in bf16 needs ~20GB+ VRAM; use the GPTQ-Int4
  or AWQ checkpoints if you're VRAM constrained). This cannot run
  inside a sandboxed, GPU-less, no-internet environment — run it on
  your own machine or a cloud GPU instance.
- `parse_letter_answer()` in `base_model.py` is a best-effort regex
  parser for "A/B/C/D" style responses. If a model tends to answer in
  full sentences instead of a letter, either tighten the prompt for
  that model or extend the parser (e.g. fuzzy-match against the choice
  text) in that model's own `predict()`.
- `dataset.missing_files()` is run automatically at the start of every
  `evaluate.py` call so a broken `--base_dir` path fails fast instead
  of silently scoring near 0%.
