# ml_model/audio_features.py
import librosa

def detect_genre(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        if len(y) == 0:
            print("🎵 Género detectado: -1 (unknown)")
            return -1  # unknown

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = sum(abs(y)) / len(y)

        if tempo == 0 or energy < 0.01:
            print("🎵 Género detectado: -1 (unknown)")
            return -1  # unknown

        if tempo > 140 and energy > 0.05:
            print("🎵 Género detectado: 2 (electronic)")
            return 2
        elif tempo < 90:
            print("🎵 Género detectado: 3 (ambient)")
            return 3
        elif energy > 0.07:
            print("🎵 Género detectado: 1 (rock)")
            return 1
        else:
            print("🎵 Género detectado: 0 (pop)")
            return 0

    except Exception as e:
        print(f"⚠️ Error al detectar género: {e}")
        return -1  # unknown


        
def detect_beats(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return tempo, beat_times.tolist()
