# preprocessing/rhythm_detection.py
import librosa

def detect_rhythm(audio_path):
    """
    Detecta el tempo y las marcas de ritmo (beats) de un archivo de audio.

    Args:
        audio_path (str): Ruta al archivo de audio.

    Returns:
        tuple: tempo estimado (BPM), lista de tiempos de beat en segundos.
    """
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return tempo, beat_times.tolist()

# Ejemplo de uso
if __name__ == "__main__":
    test_audio = "../assets/ejemplo.mp3"
    tempo, beats = detect_rhythm(test_audio)
    print(f"Tempo: {tempo:.2f} BPM")
    print(f"Beats detectados: {len(beats)}")

