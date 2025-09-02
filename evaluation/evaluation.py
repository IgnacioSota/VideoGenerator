# evaluation.py
import os
from collections import Counter

def compute_scene_diversity(clips, scene_list):
    used_scenes = set((c.source_path, round(c.start, 2), round(c.end, 2)) for c in clips)
    return len(used_scenes) / len(scene_list) if scene_list else 0

def compute_transition_stats(clips):
    transitions = [getattr(c, "transition", "none") for c in clips]
    counter = Counter(transitions)
    diversity = len(counter)
    return counter, diversity

def compute_beat_coverage(clips, beat_times):
    covered_beats = 0
    used = [False] * len(beat_times)
    for clip in clips:
        for i, beat in enumerate(beat_times):
            if clip.start <= beat < clip.end and not used[i]:
                used[i] = True
                covered_beats += 1
    return covered_beats / len(beat_times) if beat_times else 0

def compute_clip_density(clips, audio_duration):
    total_clip_time = sum([clip.duration for clip in clips])
    return total_clip_time / audio_duration if audio_duration > 0 else 0

def save_metrics_report(
    coverage,
    density,
    num_clips,
    audio_duration,
    scene_diversity,
    transition_counter,
    transition_diversity,
    output_path="output/metrics.txt"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" MÉTRICAS DE EVALUACIÓN\n")
        f.write(f" Cobertura de beats: {coverage:.2f}\n")
        f.write(f" Densidad de clips: {density:.2f}\n")
        f.write(f" Número de clips: {num_clips}\n")
        f.write(f" Duración del audio: {audio_duration:.2f} s\n")
        f.write(f" Diversidad de escenas: {scene_diversity*100:.2f}%\n")
        f.write(f" Diversidad de transiciones: {transition_diversity}\n")
        f.write(" Uso de transiciones:\n")
        for t, count in transition_counter.items():
            f.write(f"  - {t}: {count} veces\n")

