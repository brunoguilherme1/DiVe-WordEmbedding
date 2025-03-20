import joblib
import numpy as np

# Caminho do modelo treinado
MODEL_PATH = "app/iris_model.pkl"

# Carregar o modelo treinado
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print(f"Erro: O arquivo {MODEL_PATH} não foi encontrado. Certifique-se de rodar 'train.py' antes de iniciar a API.")

def predict(features):
    """Recebe uma lista de features e retorna a previsão do modelo"""
    features = np.array(features).reshape(1, -1)
    return int(model.predict(features)[0])

