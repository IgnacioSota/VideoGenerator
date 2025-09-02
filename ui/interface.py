# ui/interface.py
import tkinter as tk
from tkinter import filedialog, messagebox
import os

from preprocessing.scene_detection import detect_scenes
from preprocessing.rhythm_detection import detect_rhythm
from ml_model.video_composer import compose_video_from_scenes
from ml_model.audio_features import detect_genre

video_paths = []
audio_path = None

def select_videos():
    global video_paths
    files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mov *.avi *mkv")])
    video_paths = list(files)
    video_label.config(text=f"{len(video_paths)} vídeos seleccionados")

def select_audio():
    global audio_path
    file = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
    audio_path = file
    audio_label.config(text=os.path.basename(audio_path))

def run_pipeline(root):
    if not video_paths:
        messagebox.showerror("Error", "Selecciona al menos un vídeo")
        return
    if not audio_path:
        messagebox.showerror("Error", "Selecciona un archivo de audio")
        return

    try:
        tempo, beat_times = detect_rhythm(audio_path)
        genre = detect_genre(audio_path)

        scene_list = []
        for video_path in video_paths:
            scenes = detect_scenes(video_path)
            scene_list.extend([(video_path, start, end) for (start, end) in scenes])


        output_path = compose_video_from_scenes(
            scene_list=scene_list,
            beat_times=beat_times,
            genre = genre,
            audio_path=audio_path,
            output_path="output/final_video.mp4"
        )

        messagebox.showinfo("Éxito", f"Vídeo generado con éxito:\n{output_path}")
        root.destroy()  # cerrar ventana

    except Exception as e:
        messagebox.showerror("Error en pipeline", str(e))

def run_interface():
    global video_label, audio_label

    root = tk.Tk()
    root.title("Generador de Vídeo con Audio")
    root.geometry("400x250")

    tk.Label(root, text="Carga tus archivos:", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="Seleccionar Vídeo(s)", command=select_videos).pack(pady=5)
    video_label = tk.Label(root, text="Ningún vídeo seleccionado")
    video_label.pack()

    tk.Button(root, text="Seleccionar Audio", command=select_audio).pack(pady=5)
    audio_label = tk.Label(root, text="Ningún audio seleccionado")
    audio_label.pack()

    tk.Button(root, text="Generar Vídeo", command=lambda: run_pipeline(root), bg="green", fg="white").pack(pady=20)

    root.mainloop()
