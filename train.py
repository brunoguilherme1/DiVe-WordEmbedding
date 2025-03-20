from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Carregar os dados
iris = load_iris()
X, y = iris.data, iris.target

# Treinar o modelo
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Salvar o modelo treinado
joblib.dump(model, "app/iris_model.pkl")

print("Modelo treinado e salvo em app/iris_model.pkl")

