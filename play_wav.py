import os
import sounddevice as sd
import soundfile as sf

def play_wav(audio_dir, label):
    filename = label + ".wav"
    path = os.path.join(audio_dir, filename)

    print("Playing:", filename)

    data, samplerate = sf.read(path)
    sd.play(data, samplerate)