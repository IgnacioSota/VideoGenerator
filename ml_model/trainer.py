# ml_model/trainer.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_PATH = "ml_model/video_timing_model.pkl"
SCALER_PATH = "ml_model/video_timing_scaler.pkl"

def extract_feature_vector(interval_duration, scene_duration, scene_pos, beat_pos, motion, transition_score, beats_grouped, genre):
    return np.array([
        interval_duration,
        scene_duration,
        scene_pos,
        beat_pos,
        motion,
        transition_score,
        beats_grouped,
        genre
    ])

def train_timing_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH) #Envía el modelo a un archivo
    joblib.dump(scaler, SCALER_PATH)
    print("Modelo de sincronización entrenado y guardado.")

def predict_score_for_batch(X):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Modelo o scaler no encontrado. Entrena primero.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)

# Ejemplo de entrenamiento:
if __name__ == "__main__":
    # Generación de ejemplos ficticios
    X = []
    y = []
    for interval_dur in [0.5, 1.0, 1.5]:
        for scene_dur in [0.5, 1.0, 1.5]:
            for i in range(3):
                scene_pos = i / 3.0
                beat_pos = i / 3.0
                genre = 0
                transition_score = 1
                beats_grouped = 1
                features = extract_feature_vector(interval_dur, scene_dur, scene_pos, beat_pos, genre, transition_score, beats_grouped,genre)
                score = 1.0 - abs(interval_dur - scene_dur)  # mayor score si duran parecido
                X.append(features)
                y.append(score)
    train_timing_model(np.array(X), np.array(y))

