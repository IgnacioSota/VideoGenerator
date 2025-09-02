# generate_training_data.py
import os
import numpy as np
import librosa
import cv2
import moviepy.editor as mp
from tqdm import tqdm
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from trainer import train_timing_model
from audio_features import detect_genre, detect_beats

DATA_DIR = "training_videos"
OUTPUT_X = "ml_model/X_train.npy"
OUTPUT_Y = "ml_model/y_train.npy"

MOTION_MAX = 30.0
TRANSITION_MAX = 30.0
GENRE_CLASSES = [-1, 0, 1, 2, 3]  # unknown, pop, rock, electronic, ambient

def genre_to_onehot(genre):
    """Convierte género entero a vector one-hot de 5 columnas"""
    onehot = [0] * len(GENRE_CLASSES)
    try:
        idx = GENRE_CLASSES.index(int(genre))
    except Exception:
        idx = 0  # unknown
    onehot[idx] = 1
    return onehot

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    scenes = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
    video_manager.release()
    return scenes

def calculate_motion(video_clip, t1, t2):
    try:
        frame1 = video_clip.get_frame(t1)
        frame2 = video_clip.get_frame(t2)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(diff))
    except:
        return 0.0

def process_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio_tmp = "temp_audio.wav"
    video.audio.write_audiofile(audio_tmp, verbose=False, logger=None)

    try:
        scenes = detect_scenes(video_path)
        _, beat_times = detect_beats(audio_tmp)
        genre = detect_genre(audio_tmp)  # -1,0,1,2,3
    finally:
        try:
            os.remove(audio_tmp)
        except:
            pass

    X, y = [], []
    video_duration = max(video.duration, 1e-6)
    audio_duration = beat_times[-1] if beat_times else video_duration

    if not beat_times or len(beat_times) < 2:
        beat_times = [0.0, video_duration]

    beat_intervals = [(beat_times[i], beat_times[i+1]) for i in range(len(beat_times)-1)]
    if len(beat_times) >= 2:
        beat_durations = [beat_times[i+1] - beat_times[i] for i in range(len(beat_times)-1)]
        avg_beat_duration = float(np.mean(beat_durations)) if beat_durations else 0.5
    else:
        avg_beat_duration = 0.5

    genre_onehot = genre_to_onehot(genre)

    for interval_start, interval_end in beat_intervals:
        interval_duration = max(interval_end - interval_start, 1e-6)
        beat_pos = float(np.clip(interval_start / max(audio_duration, 1e-6), 0.0, 1.0))

        for scene_start, scene_end in scenes:
            scene_duration = max(scene_end - scene_start, 1e-6)
            scene_pos = float(np.clip(scene_start / video_duration, 0.0, 1.0))

            if scene_start <= interval_start <= scene_end:
                motion = calculate_motion(video, scene_start, min(scene_end, scene_start + 0.5))
                transition_raw = calculate_motion(video, max(scene_end - 0.5, scene_start), scene_end)
                beats_grouped = max(1, round(scene_duration / avg_beat_duration)) if avg_beat_duration > 0 else 1

                
                # Normalizaciones (como ya tienes)
                dur_sim = 1.0 - min(abs(scene_duration - interval_duration) /
                                    max(scene_duration, interval_duration, 1e-6), 1.0)
                interval_rel = float(np.clip(interval_duration / 5.0, 0.0, 1.0))
                scene_rel    = float(np.clip(scene_duration    / 5.0, 0.0, 1.0))
                motion_norm     = float(np.clip(motion / MOTION_MAX, 0.0, 1.0))
                transition_norm = float(np.clip(transition_raw / TRANSITION_MAX, 0.0, 1.0))
                beats_grouped_norm = float(beats_grouped / (1.0 + beats_grouped))
                
                # 1) Alineación posicional (más peso a beat_pos/scene_pos)
                pos_align = 1.0 - abs(scene_pos - beat_pos)           # [0,1] cuanto más cerca, mejor
                
                # 2) Balance de escalas de duración (favorece interval/scene similares también en escala)
                length_balance = 1.0 - abs(interval_rel - scene_rel)  # [0,1]
                
                # 3) Suavizar motion/transition para que no dominen
                motion_bal     = motion_norm ** 0.35      # más <1 => más comprimido
                transition_bal = transition_norm ** 0.50  # compresión moderada
                
                # Target score reequilibrado (menos motion/transition, más duraciones y posiciones)
                target_score = (
                    0.20 * dur_sim +         # similitud de duraciones
                    0.25 * pos_align +       # alineación beat-escena
                    0.20 * length_balance +  # escala de duraciones compatible
                    0.20 * transition_bal +  # transición (suavizada)
                    0.15 * motion_bal        # movimiento (suavizado)
                )


                # Features continuas + one-hot del género
                features = [
                    interval_rel, scene_rel, scene_pos, beat_pos,
                    motion_norm, transition_norm, beats_grouped_norm
                ] + genre_onehot

                X.append(features)
                y.append(target_score)

    video.close()
    return X, y

def generate_dataset():
    all_X, all_y = [], []

    for file in tqdm(os.listdir(DATA_DIR)):
        if not file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            continue
        video_path = os.path.join(DATA_DIR, file)
        try:
            X, y = process_video(video_path)
            all_X.extend(X)
            all_y.extend(y)
        except Exception as e:
            print(f"Error procesando {file}: {e}")

    if not all_X:
        raise RuntimeError("No se generaron muestras. Revisa beats/escenas.")

    X_array = np.asarray(all_X, dtype=np.float32)
    y_array = np.asarray(all_y, dtype=np.float32)

    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)
    np.save(OUTPUT_X, X_array)
    np.save(OUTPUT_Y, y_array)
    print(f"Datos guardados: {X_array.shape[0]} muestras, {X_array.shape[1]} features.")
    train_timing_model(X_array, y_array)

if __name__ == "__main__":
    generate_dataset()
