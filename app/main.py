from fastapi import FastAPI
from pydantic import BaseModel
import app.model as model

# Criando a aplicação FastAPI
app = FastAPI()

# Classe de entrada para a API
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "API FastAPI rodando no GKE!"}

@app.post("/predict/")
def predict(data: InputData):
    prediction = model.predict(data.features)
    return {"prediction": prediction}

