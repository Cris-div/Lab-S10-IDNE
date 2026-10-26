import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Datos proporcionados en el laboratorio. 0 = no fuga; 1 = fuga.
datos = pd.DataFrame({
    "edad": [22, 25, 28, 31, 35, 38, 40, 43, 46, 47, 50, 52, 53, 55, 56, 57, 60, 62, 29, 33],
    "ingresos": [1200, 1500, 1800, 1900, 2500, 2700, 3000, 3300, 3500, 3200, 4100, 4400, 4300, 4800, 5000, 5100, 5200, 5400, 1600, 2100],
    "deuda": [500, 700, 900, 950, 1200, 1300, 1600, 1700, 1800, 1500, 2300, 2200, 2400, 2800, 3000, 2900, 3100, 3300, 800, 1000],
    "score_crediticio": [650, 620, 610, 605, 590, 585, 570, 555, 560, 580, 530, 540, 520, 510, 500, 505, 490, 480, 615, 600],
    "fuga": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
})
X = datos.drop(columns="fuga")
y = datos["fuga"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=29, stratify=y
)
# La media y la desviación se aprenden únicamente del entrenamiento.
# Escalar las variables antes de entrenar el modelo.
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = SVC(kernel="linear", C=1)
modelo.fit(X_train_escalado, y_train)
predicciones = modelo.predict(X_test_escalado)

print("EJERCICIO 1 - FUGA DE CLIENTES")
print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predicciones):.4f}")
print(pd.DataFrame({"fuga_real": y_test, "fuga_predicha": predicciones}).to_string())
