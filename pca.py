# pca.py
# ------
# PCA plotting module: projects CLAP embeddings from 512D → 2D and draws
# them as a scatter plot. Also wires up click-to-play via play_wav.py.
# Kept separate from clap_demo_main.py so the AI algorithm is front-and-centre
# during a presentation without the matplotlib plumbing getting in the way.

import numpy as np                  # Numerical arrays — all embeddings live here
import matplotlib.pyplot as plt     # For drawing the scatter plot
from sklearn.decomposition import PCA   # Dimensionality reduction: 512D → 2D

from play_wav import play_wav       # Our isolated WAV playback helper


def build_pca(audio_embeddings):
    """
    Fit a PCA model on the audio embeddings and return it.

    We fit on audio embeddings only (not text) so the coordinate system is
    stable across multiple queries — the audio dots won't jump around each
    time you type a new query. Text embeddings are projected into this same
    coordinate system afterward using pca.transform().

    Parameters:
      audio_embeddings : (N, 512) NumPy array — one row per audio clip

    Returns:
      pca : fitted sklearn PCA object (n_components=2)
    """

    # PCA (Principal Component Analysis) finds the two directions in 512D space
    # that capture the most variance across all audio embeddings.
    # Think of it as finding the "widest" and "second-widest" axes of the cloud
    # of points, then looking at the cloud from that angle — the projection that
    # shows the most spread in the data.
    pca = PCA(n_components=2)   # Keep only the top 2 principal components
    pca.fit(audio_embeddings)   # Learn the 2 principal axes from the audio data
    return pca


def plot_pca_vectors(pca, audio_embeddings, text_emb, labels, sims, query, audio_dir):
    """
    Draw the PCA scatter plot for one query and wire up click-to-play.

    Parameters:
      pca              : fitted sklearn PCA object from build_pca()
      audio_embeddings : (N, 512) array — all audio embeddings
      text_emb         : (1, 512) array — the current text query embedding (unused for display)
      labels           : list of N strings — display names for each clip
      sims             : (N,) array — cosine similarity of each clip to the query
      query            : str — the raw query text (unused for display)
      audio_dir        : str — path to the clips folder (needed for click-to-play)
    """

    # Find the index of the audio clip with the highest cosine similarity
    # to the query — this is the "closest" match.
    closest_idx = int(np.argmax(sims))

    # Project all 512D embeddings down to 2D using the axes learned above.
    # pca.transform() applies the matrix multiplication: (N, 512) @ (512, 2) → (N, 2)
    audio_2d = pca.transform(audio_embeddings)  # Shape: (N, 2)

    # Create the figure and axes. figsize is in inches at 100 dpi default.
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")    # Figure background (outside axes area)
    ax.set_facecolor("white")           # Axes background

    # Draw faint crosshair lines through the origin (0, 0).
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)

    # --- Draw one dot per audio clip ---
    for i, (pt, label) in enumerate(zip(audio_2d, labels)):

        # Highlight the closest match in green; gray for the rest.
        color = "#10a30d" if i == closest_idx else "#64748b"
        size  = 40        if i == closest_idx else 20

        # ax.scatter plots a single dot at the embedding's 2D position.
        # zorder=3 ensures dots render on top of the crosshair lines.
        ax.scatter(pt[0], pt[1], color=color, s=size, zorder=3)

        # Label each dot with the clip name, offset slightly so it
        # doesn't overlap the marker.
        ax.text(pt[0] + 0.01, pt[1] + 0.01, label,
                fontsize=8, color=color)

    ax.set_title("2D Projection of CLAP Embeddings")

    ax.set_xlabel("PC 1")   # First principal component (most variance)
    ax.set_ylabel("PC 2")   # Second principal component (second-most variance)

    # Auto-scale the axis limits to fit all points with a small margin.
    pad = 0.05
    xvals = audio_2d[:, 0]
    yvals = audio_2d[:, 1]
    ax.set_xlim(xvals.min() - pad, xvals.max() + pad)
    ax.set_ylim(yvals.min() - pad, yvals.max() + pad)

    # --------------------------------------------------------
    # CLICK-TO-PLAY HANDLER
    # --------------------------------------------------------
    def on_click(event):
        """
        Called by Matplotlib whenever the user clicks on the figure.
        Finds the nearest audio embedding in 2D plot space and plays it
        via the OS default audio player.
        """

        # Ignore clicks outside the axes area (e.g. on the toolbar or title).
        if event.inaxes != ax:
            return

        # event.xdata / event.ydata are the click coordinates in data space
        # (i.e. the same coordinate system as the plotted points).
        x_click = event.xdata
        y_click = event.ydata

        # Compute Euclidean distance from the click to every audio point in 2D.
        # This is standard √((x₁-x₂)² + (y₁-y₂)²) applied to all points at once
        # using NumPy broadcasting — no Python loop needed.
        dists = np.sqrt((audio_2d[:, 0] - x_click) ** 2 +
                        (audio_2d[:, 1] - y_click) ** 2)

        # The closest point in 2D plot space.
        idx = np.argmin(dists)

        # Only trigger playback if the click landed within 0.15 plot units
        # of an actual point — prevents accidental playback from stray clicks.
        if dists[idx] > 0.15:
            return

        # Delegate to play_wav.py — all subprocess plumbing lives there.
        play_wav(audio_dir, labels[idx])

    # Register on_click as the callback for mouse button press events.
    # Matplotlib's event system will call it automatically on each click.
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Adjusts subplot margins so labels aren't clipped at the figure edge.
    plt.tight_layout()

    # Display the window. This call blocks until the window is closed,
    # which is why the query loop pauses while the plot is open.
    plt.show()
