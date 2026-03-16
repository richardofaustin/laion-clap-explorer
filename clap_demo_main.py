# clap_demo_main.py
# -------------------------------------------------
# WHAT THIS PROGRAM DOES:
# CLAP (Contrastive Language-Audio Pretraining) is a neural network trained to
# understand the relationship between sounds and natural language. It was trained
# by showing it millions of (audio clip, text description) pairs and teaching it
# to recognize which pairs go together — similar to how CLIP works for images.
#
# The result is a shared "embedding space": a high-dimensional coordinate system
# (512 dimensions) where audio clips and text phrases that mean similar things
# end up pointing in similar directions from the origin. This program lets you
# type a text query, compute its location in that space, and find which audio
# clip lives closest to it. PCA then squashes 512 dimensions down to 2 so we
# can draw the whole thing as an arrow diagram.
#
# Companion modules:
#   pca.py      — fits PCA and draws the vector diagram
#   play_wav.py — plays a WAV file via the OS default audio player

import os           # For building file paths that work on any OS
import torch        # PyTorch — the deep learning framework CLAP is built on
import librosa      # Audio loading library; handles resampling and format conversion
import numpy as np  # Numerical arrays — all embeddings live here after leaving PyTorch
from transformers import ClapProcessor, ClapModel       # Hugging Face wrappers for the CLAP model
from sklearn.metrics.pairwise import cosine_similarity  # Measures angle between two vectors

from pca import build_pca, plot_pca_vectors  # Our PCA + plotting helpers

# ============================================================
# DEVICE SETUP
# ============================================================
# PyTorch can run on the CPU or on an NVIDIA GPU (CUDA).
# GPU inference is ~10-50x faster for large models like CLAP.
# torch.cuda.is_available() returns True if a compatible GPU is present.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# LOAD THE CLAP MODEL
# ============================================================
# CLAP has two sub-networks (encoders):
#   1. An AUDIO encoder  — based on HTS-AT, a hierarchical audio transformer
#   2. A TEXT encoder    — based on RoBERTa, a variant of BERT
#
# Both encoders were trained jointly using contrastive loss: the model was
# rewarded when it placed matching (audio, text) pairs close together in the
# shared embedding space, and penalized when unrelated pairs ended up close.
# After training, the two encoders share the same coordinate system, so a text
# description and an audio clip can be directly compared by measuring the angle
# between their vectors.
#
# "laion/clap-htsat-unfused" is the model ID on Hugging Face Hub.
# "unfused" means the audio and text towers were trained separately before
# being joined — as opposed to a fused architecture where weights are shared.
#model_name = "laion/clap-htsat-unfused"
model_name = "laion/larger_clap_music_and_speech"

# ClapProcessor handles all pre-processing for both modalities:
#   - For audio: converts raw waveforms into log-mel spectrograms
#     (a 2D image-like representation of frequency over time) that the
#     audio encoder expects.
#   - For text: tokenizes strings into integer IDs that the text encoder expects.
# from_pretrained() downloads the config and weights from Hugging Face on first run,
# then caches them locally for subsequent runs.
processor = ClapProcessor.from_pretrained(model_name)

# ClapModel wraps both encoders plus a learned linear projection layer that
# maps each encoder's output into the final shared 512-dimensional space.
model = ClapModel.from_pretrained(model_name)

# Move all model weights to the chosen device (GPU memory or CPU RAM).
model.to(device)

# Switch model to inference mode. This disables dropout layers (used during
# training to prevent overfitting) and tells batch-norm layers to use running
# statistics rather than per-batch statistics — both of which produce
# deterministic, stable embeddings at inference time.
model.eval()

# ============================================================
# LOAD AUDIO CLIPS AND COMPUTE THEIR EMBEDDINGS
# ============================================================
# Build an absolute path to the "clips" folder sitting next to this script,
# so the program works regardless of where it's launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "clips")

# Collect all .wav filenames from the clips directory, sorted alphabetically
# so the order is consistent across platforms.
files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")])

if not files:
    raise ValueError("No .wav files found in clips folder.")

print("Found audio files:", files)

# These lists will accumulate one embedding array and one label per audio file.
audio_embeddings = []
labels = []

for f in files:
    # Build the full file path for this clip.
    path = os.path.join(AUDIO_DIR, f)

    # librosa.load() reads the audio file and resamples it to 48 000 Hz
    # (the sample rate CLAP's audio encoder was trained on).
    # mono=True mixes stereo down to a single channel.
    # The result `audio` is a 1D NumPy array of float32 amplitude values.
    audio, sr = librosa.load(path, sr=48000, mono=True)

    # The processor converts the raw waveform into a log-mel spectrogram:
    # it applies a Short-Time Fourier Transform (STFT) to get frequency content
    # over time, maps the frequency bins onto the mel scale (which matches human
    # pitch perception), and takes the log of the magnitudes.
    # return_tensors="pt" means "give me PyTorch tensors, not NumPy arrays."
    inputs = processor(
        audio=audio,
        sampling_rate=48000,
        return_tensors="pt"
    )

    # Move each input tensor to the same device as the model.
    # The dict comprehension iterates over the processor's output keys
    # (e.g. "input_features") and calls .to(device) on each tensor.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # torch.no_grad() disables gradient tracking for everything inside the block.
    # Gradients are needed for training (backpropagation) but waste memory and
    # computation during inference, so we turn them off here.
    with torch.no_grad():
        # Pass the spectrogram through CLAP's audio encoder + projection layer.
        # Internally the HTS-AT transformer divides the spectrogram into patches,
        # processes them through ~12 attention layers, and produces a single
        # 512-dimensional summary vector for the whole clip.
        outputs = model.get_audio_features(**inputs)

    # Different versions of the Hugging Face CLAP wrapper name the output
    # attribute differently. We check for both names so the script is robust.
    if hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output   # Older API: the pooled CLS token output
    elif hasattr(outputs, "audio_embeds"):
        emb = outputs.audio_embeds    # Newer API: the projected audio embedding
    else:
        raise RuntimeError("Unknown output structure from get_audio_features()")

    # *** THE KEY CLAP IDEA: L2 NORMALIZATION ***
    # Dividing the embedding vector by its own magnitude (norm) scales it onto
    # the surface of a unit hypersphere — every vector now has length exactly 1.
    # Why does this matter?
    #   cosine similarity = dot product of two unit vectors = cos(angle between them)
    # After normalization, comparing two embeddings is just a dot product, and
    # the result directly tells us the angle between them in 512D space.
    # Vectors pointing in the same direction → angle ≈ 0° → similarity ≈ 1.0
    # Vectors pointing at 90°               → orthogonal → similarity ≈ 0.0
    # Vectors pointing opposite             → angle = 180° → similarity ≈ -1.0
    emb = emb / emb.norm(dim=-1, keepdim=True)

    # .detach() removes the tensor from PyTorch's computation graph (no longer
    # needed after inference). .cpu() moves it back to RAM. .numpy() converts
    # to a NumPy array for use with sklearn and matplotlib.
    audio_embeddings.append(emb.detach().cpu().numpy())

    # Store the filename without the .wav extension as a human-readable label.
    labels.append(f.replace(".wav", ""))

# Stack the list of (1, 512) arrays into a single (N, 512) matrix,
# where N is the number of audio clips. Each row is one audio embedding.
audio_embeddings = np.vstack(audio_embeddings)

print("\nAudio embeddings computed.\n")

# Fit PCA once on the audio embeddings so the plot layout stays stable
# across multiple queries. See pca.py for a full explanation.
pca = build_pca(audio_embeddings)

# ============================================================
# INTERACTIVE QUERY LOOP
# ============================================================
# Keep prompting for text queries until the user types "exit".
while True:

    query = input("Type a query (or 'exit'): ")

    if query.lower() == "exit":
        break

    # --- Encode the text query through CLAP's text encoder ---

    # The processor tokenizes the query string:
    # it splits it into subword tokens (e.g. "bark" → ["bar", "##k"]),
    # maps each token to an integer ID from CLAP's vocabulary,
    # and pads/truncates to a fixed sequence length.
    # return_tensors="pt" gives us PyTorch tensors ready for the model.
    text_inputs = processor(text=[query], return_tensors="pt")

    # Move the token ID tensors to the same device as the model.
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        # Run the token IDs through CLAP's text encoder (RoBERTa-based transformer)
        # + the shared projection layer to get a 512D text embedding.
        # The transformer uses self-attention to build a context-aware representation
        # of the query — "dog bark" is processed differently than "bark tree" because
        # the surrounding tokens influence each token's representation.
        text_outputs = model.get_text_features(**text_inputs)

    # Same dual-API check as for audio embeddings above.
    if hasattr(text_outputs, "pooler_output"):
        text_emb = text_outputs.pooler_output  # Older API
    elif hasattr(text_outputs, "text_embeds"):
        text_emb = text_outputs.text_embeds    # Newer API
    else:
        raise RuntimeError("Unknown output structure from get_text_features()")

    # L2-normalize the text embedding onto the unit hypersphere,
    # exactly as we did for audio embeddings. This is critical:
    # without normalization, longer texts (larger raw magnitude) would
    # appear more similar to everything, distorting the rankings.
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Detach from the computation graph and move to CPU NumPy for sklearn.
    text_emb = text_emb.detach().cpu().numpy()

    # --- Compute cosine similarity between the query and all audio clips ---
    # cosine_similarity takes two 2D arrays: (1, 512) text vs (N, 512) audio.
    # For unit vectors, cosine similarity = dot product = cos(angle between them).
    # The result is a (1, N) array; [0] unwraps it to a 1D array of N scores.
    # Score of 1.0 → identical direction (perfect match)
    # Score of 0.0 → orthogonal (unrelated)
    # Score of -1.0 → opposite direction (semantic opposites)
    sims = cosine_similarity(text_emb, audio_embeddings)[0]

    # Pair each label with its similarity score and sort highest-first
    # so we can print a ranked list for the user.
    ranked = sorted(zip(labels, sims),
                    key=lambda x: x[1], reverse=True)

    print("\nSimilarity ranking:")
    for name, score in ranked:
        # :15s left-pads the name to 15 characters for clean column alignment.
        print(f"{name:15s} {score:.3f}")
    print()

    # Draw the PCA vector diagram for this query.
    # The function blocks here until the plot window is closed.
    plot_pca_vectors(pca, audio_embeddings, text_emb, labels, sims, query, AUDIO_DIR)
