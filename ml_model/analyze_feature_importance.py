import joblib
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo entrenado
MODEL_PATH = "ml_model/video_timing_model.pkl"
model = joblib.load(MODEL_PATH)

# Cargar datos de entrenamiento para obtener los nombres y forma
X = np.load("ml_model/X_train.npy")

# Nombres de las features en el orden correcto
feature_names = [
    "interval_duration",
    "scene_duration",
    "scene_pos",
    "beat_pos",
    "motion",
    "transition_score",
    "beats_grouped",
    "genre_unknown",
    "genre_pop",
    "genre_rock",
    "genre_electronic",
    "genre_ambient"
]

# Verificar si el modelo tiene el atributo
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_

    print("📊 Importancia de las variables:")
    for name, importance in zip(feature_names, importances):
        print(f"{name:<20}: {importance:.4f}")

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Importancia")
    plt.title("Importancia de las variables del modelo")
    plt.tight_layout()
    plt.show()
else:
    print("❌ El modelo cargado no tiene el atributo 'feature_importances_'. ¿Estás seguro de que es un RandomForestRegressor?")
