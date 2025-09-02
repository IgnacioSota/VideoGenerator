# preprocessing/scene_detection.py
import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path, threshold=30.0):
    """
    Detecta escenas en un vídeo usando PySceneDetect.

    Args:
        video_path (str): Ruta del archivo de vídeo.
        threshold (float): Sensibilidad de detección (mayor = menos cortes).

    Returns:
        list of tuples: Lista de escenas como pares (inicio, fin) en segundos.
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    # Convertir a segundos para facilitar el procesamiento posterior
    scenes_sec = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]

    video_manager.release()
    return scenes_sec

# Ejemplo de uso:
if __name__ == "__main__":
    test_video = "../assets/ejemplo.mp4"
    escenas = detect_scenes(test_video)
    for i, (start, end) in enumerate(escenas):
        print(f"Escena {i+1}: {start:.2f}s - {end:.2f}s")

