import pandas as pd
import yt_dlp
import os

# Ruta al CSV con una ID por línea
csv_path = "video_ads.csv"

# Carpeta de destino (un directorio hacia atrás)
output_folder = os.path.join("..", "training_videos")
os.makedirs(output_folder, exist_ok=True)

# Leer el CSV sin cabecera, una sola columna
df = pd.read_csv(csv_path, header=None)

# Configuración para yt-dlp
ydl_opts = {
    'format': 'mp4',
    'outtmpl': os.path.join(output_folder, '%(id)s.%(ext)s'),
    'quiet': False,
    'ignoreerrors': True,   # continuar si un vídeo falla
}

# Descargar cada vídeo
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for video_id in df.iloc[:, 0]:
        # Quitar comillas si las hay
        video_id = str(video_id).strip().strip("'").strip('"')
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            print(f"Descargando: {url}")
            ydl.download([url])
        except Exception as e:
            print(f"Error descargando {url}: {e}")

