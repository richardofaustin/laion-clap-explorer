# play_wav.py
# -----------
# Utility module for playing a WAV file via the OS default audio player.
# Kept separate so the main CLAP demo can be presented without distraction
# from platform-specific subprocess plumbing.

import os           # For building file paths that work on any OS
import subprocess   # For launching the system audio player (Windows "start" command)


def play_wav(audio_dir, label):
    """
    Play the WAV file corresponding to `label` from `audio_dir`.

    Parameters:
      audio_dir : str — absolute path to the folder containing .wav files
      label     : str — filename stem (without .wav extension)
    """

    filename = label + ".wav"
    path = os.path.join(audio_dir, filename)

    print("Playing:", filename)

    # subprocess.Popen launches the file without blocking the Python process.
    # shell=True + "start" is the Windows way to open a file with its
    # default associated application (e.g. Windows Media Player).
    # On macOS you'd use ["open", path]; on Linux ["xdg-open", path].
    subprocess.Popen(["start", path], shell=True)
