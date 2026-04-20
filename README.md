# laion-clap-explorer

An interactive demo that uses the [LAION CLAP](https://github.com/LAION-AI/CLAP) audio-language model to search a folder of WAV files using natural language queries. Results are visualized as a 2D PCA vector diagram — click any arrow to play the corresponding audio clip.

> **Note:** The [original LAION CLAP repo](https://github.com/LAION-AI/CLAP) covers model training, architecture, and research reproduction. This repo demonstrates practical inference: how to load the model, compute embeddings, compare audio to text queries, and visualize the embedding space interactively.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

CLAP (Contrastive Language-Audio Pretraining) is a neural network trained on millions of (audio, text) pairs. It learns a shared 512-dimensional embedding space where sounds and descriptions that mean similar things point in the same direction from the origin.

This demo lets you:
- Load a folder of WAV files and compute their CLAP audio embeddings
- Type a natural language query (e.g. `"the sound of a flute"`)
- See a ranked list of your audio clips by cosine similarity to the query
- View a 2D PCA projection of all embeddings as an arrow diagram
- Click any arrow on the plot to play that audio clip

---

## Project structure

```
laion-clap-explorer/
├── clap_demo_main.py   # Main program — CLAP model loading, embedding, similarity ranking
├── pca.py              # PCA fitting and matplotlib vector diagram
├── play_wav.py         # WAV playback via OS default audio player (Windows)
├── clips/              # Put your .wav files here
├── pyproject.toml      # Dependencies for uv
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- [ffmpeg](https://ffmpeg.org/) (for converting audio files to the required format)
- Windows (for WAV playback via `start`; see [Other platforms](#other-platforms) below)
- A CUDA-capable GPU is recommended but not required

---

## Installation

### 1. Install uv. It is used to install the dependencies.

```cmd
Windows PowerShell
PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
Linux/MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a project folder and clone the repo into it.

```cmd
git clone https://github.com/richardofaustin/laion-clap-explorer.git
```

### 3. Install dependencies

```cmd
cd laion-clap-explorer
uv sync
```

`uv sync` reads `pyproject.toml` and installs all required packages into a local virtual environment automatically — no separate `venv` or `pip install` step needed.

> On first run, the CLAP model weights (~600MB) will be downloaded automatically from Hugging Face and cached locally.

---

### 4. Running the demo

```cmd
uv run python clap_demo_main.py
```

Then type a natural language query at the prompt:

```
Type a query (or 'exit'): the sound of a flute
```

**Tips for better results:**
- Use descriptive phrases rather than single words: `"the sound of a piano"` works better than `"piano"`
- Include acoustic context: `"breathy flute melody"`, `"plucked acoustic guitar"`, `"jazz drum kit"`
- CLAP was trained on caption-style descriptions, not synthesis terminology — avoid terms like `"sawtooth wave"` or `"square wave"`; use `"buzzy electronic tone"` instead

---

## Other platforms

WAV playback in `play_wav.py` uses the Windows `start` command. To use on other platforms, change the `subprocess.Popen` line:

**macOS:**
```python
subprocess.Popen(["open", path])
```

**Linux:**
```python
subprocess.Popen(["xdg-open", path])
```

---

## Model

This demo uses [`laion/larger_clap_music_and_speech`](https://huggingface.co/laion/larger_clap_music_and_speech), a CLAP checkpoint fine-tuned on music and speech data. It performs better on musical instruments than the base `clap-htsat-unfused` model.

To swap models, change this line in `clap_demo_main.py`:

```python
model_name = "laion/larger_clap_music_and_speech"
```

---

## References

- **CLAP paper:** [Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) — Wu et al., 2022
- **LAION CLAP GitHub:** https://github.com/LAION-AI/CLAP
- **Hugging Face model card:** https://huggingface.co/laion/larger_clap_music_and_speech
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers
- **librosa:** https://librosa.org
- **scikit-learn PCA:** https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

---

## License

MIT — see [LICENSE](LICENSE)
