# ml_model/video_composer.py
import moviepy.editor as mp
import os
from moviepy.editor import AudioFileClip
import cv2
import numpy as np
# Nota: ya no dependemos de extract_feature_vector; generamos aquí el vector como en training
from ml_model.trainer import predict_score_for_batch
from evaluation.evaluation import (
    compute_beat_coverage, compute_clip_density, save_metrics_report,
    compute_scene_diversity, compute_transition_stats
)
from tqdm import tqdm
import random

# ----------------------------
# Parámetros/ayudas de normalización (igual que en training)
# ----------------------------
MOTION_MAX = 30.0
TRANSITION_MAX = 30.0
GENRE_CLASSES = [-1, 0, 1, 2, 3]  # unknown, pop, rock, electronic, ambient

def genre_to_onehot(genre):
    onehot = [0] * len(GENRE_CLASSES)
    try:
        idx = GENRE_CLASSES.index(int(genre))
    except Exception:
        idx = 0  # unknown
    onehot[idx] = 1
    return onehot

def build_feature_vector(interval_duration, scene_duration, scene_pos, beat_pos,
                         motion, transition_score, beats_grouped, genre):
    """
    Devuelve el vector de features normalizadas con el MISMO formato que training:
      [interval_rel, scene_rel, scene_pos, beat_pos, motion_norm, transition_norm, beats_grouped_norm] + genre_onehot(5)
    Además, retorna beats_grouped_norm por separado para re-pesado.
    """
    # Escalas relativas de duración (cap a 5s como en training)
    interval_rel = float(np.clip(interval_duration / 5.0, 0.0, 1.0))
    scene_rel    = float(np.clip(scene_duration    / 5.0, 0.0, 1.0))

    # Posiciones en [0,1]
    scene_pos = float(np.clip(scene_pos, 0.0, 1.0))
    beat_pos  = float(np.clip(beat_pos,  0.0, 1.0))

    # Normalizaciones continuas
    motion_norm     = float(np.clip(motion / MOTION_MAX, 0.0, 1.0))
    transition_norm = float(np.clip(transition_score / TRANSITION_MAX, 0.0, 1.0))

    # beats_grouped sin máximo fijo: función saturante
    beats_grouped_norm = float(beats_grouped / (1.0 + beats_grouped))

    genre_onehot = genre_to_onehot(genre)

    features = [
        interval_rel, scene_rel, scene_pos, beat_pos,
        motion_norm, transition_norm, beats_grouped_norm
    ] + genre_onehot

    return features, beats_grouped_norm


def calculate_histogram_similarity(clip1, clip2):
    try:
        frame1 = clip1.get_frame(clip1.duration - 0.01)
        frame2 = clip2.get_frame(0.01)

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)

        hist1 = cv2.calcHist([frame1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity
    except:
        return 0

def calculate_motion(video_clip, t1, t2):
    try:
        frame1 = video_clip.get_frame(t1)
        frame2 = video_clip.get_frame(t2)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff)
    except:
        return 0

def apply_transition(clip, transition_type, duration):
    if transition_type == "crossfade":
        return clip.crossfadein(duration)
    elif transition_type == "fade":
        return clip.fadein(duration/2).fadeout(duration/2)
    elif transition_type == "dissolve":
        # Simula un dissolve usando crossfade más suave
        return clip.crossfadein(duration)
    elif transition_type == "flash":
        flash_duration = 0.2
        white_clip = mp.ColorClip(size=clip.size, color=(255, 255, 255), duration=flash_duration)
        white_clip = white_clip.fadein(0.1).fadeout(0.1)
        return mp.concatenate_videoclips([white_clip, clip.set_start(flash_duration)], method="compose")
    elif transition_type == "none":
        return clip
    else:
        return clip

def compose_video_from_scenes(scene_list, beat_times, genre, audio_path=None, output_path="output/final_video.mp4"):
    print("🔄 Iniciando composición del vídeo...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("📥 Cargando vídeos...")
    video_clips = {vp: mp.VideoFileClip(vp) for vp in set(v for v, _, _ in scene_list)}
    subclips = []

    max_group_size = 3
    beat_intervals = []
    for group_size in range(1, max_group_size + 1):
        for i in range(len(beat_times) - group_size):
            start = beat_times[i]
            end = beat_times[i + group_size]
            beat_intervals.append((start, end, group_size))

    print(f"🎵 {len(beat_intervals)} intervalos de beat detectados")

    last_clip = None
    feature_vectors = []
    candidates = []

    print("🧠 Calculando vectores de características para todas las combinaciones escena-intervalo...")
    for beat_idx, (interval_start, interval_end, beats_grouped) in tqdm(enumerate(beat_intervals), total=len(beat_intervals)):
        interval_duration = interval_end - interval_start
        beat_pos = interval_start / beat_times[-1] if beat_times else 0.0

        for scene_idx, (video_path, scene_start, scene_end) in enumerate(scene_list):
            video = video_clips[video_path]
            scene_duration = scene_end - scene_start
            scene_pos = scene_start / video.duration

            motion = calculate_motion(video, scene_start, min(scene_end, scene_start + 0.5))
            transition_score = calculate_motion(video, max(scene_end - 0.5, scene_start), scene_end)

            # ---- NUEVO: construir features normalizadas + one-hot de género ----
            features, beats_grouped_norm = build_feature_vector(
                interval_duration,
                scene_duration,
                scene_pos,
                beat_pos,
                motion,
                transition_score,
                beats_grouped,
                genre
            )

            feature_vectors.append(features)
            # Guardamos beats_grouped_norm para re-peso posterior
            candidates.append((beat_idx, scene_idx, interval_start, interval_end, video_path, scene_start, scene_end, beats_grouped_norm))

    print(f"📊 Total de combinaciones evaluadas: {len(feature_vectors)}")
    if not feature_vectors:
        raise ValueError("No hay pares válidos de escena-intervalo para evaluar.")

    print("🤖 Prediciendo puntuaciones de sincronización con el modelo...")
    scores = predict_score_for_batch(np.array(feature_vectors, dtype=np.float32))

    assignments = {}
    used_scene_indices = set()
    used_beats = set()
    total_duration = 0
    audio_duration = AudioFileClip(audio_path).duration if audio_path else beat_times[-1]

    print("✅ Asignando las mejores combinaciones sin repetir escenas...")
    # Re-peso por densidad rítmica normalizada (ya en [0,1))
    scored_candidates = []
    for idx, score in enumerate(scores):
        _, _, _, _, _, _, _, beats_grouped_norm = candidates[idx]
        weighted_score = float(score) + 0.2 * float(beats_grouped_norm)  # ajusta el 0.2 si lo ves
        scored_candidates.append((idx, weighted_score))

    sorted_candidates = sorted(scored_candidates, key=lambda x: -x[1])

    for idx, score in sorted_candidates:
        beat_idx, scene_idx, interval_start, interval_end, video_path, scene_start, scene_end, _ = candidates[idx]
        if beat_idx in assignments or scene_idx in used_scene_indices:
            continue

        interval_duration = interval_end - interval_start
        clip_end = min(scene_start + interval_duration, scene_end)
        if clip_end <= scene_start:
            continue

        assignments[beat_idx] = (scene_idx, interval_start, interval_end, video_path, scene_start, clip_end)
        used_scene_indices.add(scene_idx)
        used_beats.add(beat_idx)
        total_duration += (clip_end - scene_start)

        if total_duration >= audio_duration:
            break

    print(f"🎬 Generando {len(assignments)} clips seleccionados")
    print(f"📌 Escenas disponibles: {len(scene_list)} | Escenas utilizadas: {len(used_scene_indices)}")

    recent_transitions = []
    max_same_transition = 2  # Número máximo de veces que puede repetirse la misma transición seguida

    for beat_idx in sorted(assignments.keys()):
        scene_idx, interval_start, interval_end, video_path, scene_start, clip_end = assignments[beat_idx]
        video = video_clips[video_path]
        print(f"🧩 Beat {beat_idx}: escena {scene_idx}, duración clip: {clip_end - scene_start:.2f} s")
        clip = video.subclip(scene_start, clip_end)

        # Decidir tipo de transición
        transition_type = "none"
        transition_base_duration = 0.3
        
        if last_clip is not None:
            similarity = calculate_histogram_similarity(last_clip, clip)

            # Normaliza motion para consistencia con training
            motion_raw = calculate_motion(video, scene_start, min(clip_end, scene_start + 0.5))
            motion_norm = float(np.clip(motion_raw / MOTION_MAX, 0.0, 1.0))

            # Duración del beat actual
            current_beat_duration = interval_end - interval_start
            transition_base_duration = min(max(current_beat_duration * 0.4, 0.2), 1.0)  # entre 0.2s y 1.0s

            if similarity < 0.4 or motion_norm > 0.85:
                transition_type = "flash"
            elif similarity < 0.55:
                transition_type = "dissolve"
            elif similarity < 0.7:
                transition_type = "crossfade"
            elif similarity < 0.82:
                transition_type = "fade"
            else:
                transition_type = "none" if random.random() > 0.2 else "fade"
        
            # Evitar repetir la misma transición demasiadas veces
            if recent_transitions.count(transition_type) >= max_same_transition:
                for alt in ["fade", "dissolve", "flash", "crossfade", "none"]:
                    if alt != transition_type and recent_transitions.count(alt) < max_same_transition:
                        transition_type = alt
                        break

        recent_transitions.append(transition_type)
        if len(recent_transitions) > 5:
            recent_transitions.pop(0)
            
        transitioned_clip = apply_transition(clip, transition_type, transition_base_duration)

        if transitioned_clip.audio is None and audio_path:
            transitioned_clip = transitioned_clip.set_audio(AudioFileClip(audio_path).subclip(0, transitioned_clip.duration))
        
        transitioned_clip.source_path = video_path
        transitioned_clip.transition = transition_type
        print(f"✅ Clip añadido: duración {transitioned_clip.duration:.2f} s")
        subclips.append(transitioned_clip)
        last_clip = clip

    print(f"🧮 Total de subclips generados (antes de filtrado): {len(subclips)}")

    valid_subclips = []
    for idx, clip in enumerate(subclips):
        if clip is None:
            print(f"❌ Clip {idx} es None, se ignora.")
            continue
        if clip.duration <= 0:
            print(f"❌ Clip {idx} tiene duración inválida: {clip.duration:.2f}s, se ignora.")
            continue
        print(f"🎞️ Clip {idx} válido: duración = {clip.duration:.2f}s")
        valid_subclips.append(clip)

    print(f"✅ Total de subclips válidos: {len(valid_subclips)}")
    if not valid_subclips:
        raise ValueError("❌ No se pudo ensamblar ningún clip válido para el vídeo final.")

    print("📦 Concatenando los clips finales...")
    final_clip = mp.concatenate_videoclips(valid_subclips, method="compose")

    if audio_path:
        print("🔊 Añadiendo audio al vídeo final...")
        audio = AudioFileClip(audio_path)
        audio_duration = min(audio.duration, final_clip.duration)
        final_clip = final_clip.set_audio(audio.subclip(0, audio_duration))
        final_clip = final_clip.set_duration(audio_duration)

    print(f"🎚️ Duración audio: {audio_duration:.2f}s | Duración vídeo: {final_clip.duration:.2f}s")
    print("💾 Exportando vídeo...")
    print("📈 Calculando métricas objetivas de evaluación...")
    coverage = compute_beat_coverage(valid_subclips, beat_times)
    density = compute_clip_density(valid_subclips, final_clip.duration)
    scene_div = compute_scene_diversity(valid_subclips, scene_list)
    transitions, trans_div = compute_transition_stats(valid_subclips)
    
    save_metrics_report(
        coverage,
        density,
        len(valid_subclips),
        audio_duration,
        scene_div,
        transitions,
        trans_div
    )
    
    final_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    final_clip.close()
    for clip in video_clips.values():
        clip.close()
    if audio_path:
        audio.close()

    print("✅ Vídeo generado con éxito.")
    return output_path
