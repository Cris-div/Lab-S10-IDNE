import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Datos del laboratorio. 0 = transacción normal; 1 = fraude.
datos = pd.DataFrame({
    "monto": [20, 35, 50, 80, 120, 150, 200, 250, 300, 450, 700, 900, 1200, 1500, 1800, 2200, 2500, 3000, 3500, 4000],
    "operaciones_dia": [1, 1, 2, 2, 3, 2, 4, 3, 5, 6, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30],
    "hora_transaccion": [9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6],
    "distancia_km": [1, 2, 2, 3, 5, 4, 6, 8, 10, 12, 20, 35, 50, 80, 100, 120, 150, 200, 250, 300],
    "fraude": [0] * 10 + [1] * 10,
})
X = datos.drop(columns="fraude")
y = datos["fraude"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=29, stratify=y
)
# Escalar las variables antes de entrenar el modelo.
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = SVC(kernel="linear", C=1)
modelo.fit(X_train_escalado, y_train)
predicciones = modelo.predict(X_test_escalado)
matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])

print("EJERCICIO 2 - DETECCIÓN DE FRAUDE")
print("Matriz de confusión (filas = real; columnas = predicho):")
print(pd.DataFrame(matriz, index=["Normal (0)", "Fraude (1)"],
                   columns=["Normal (0)", "Fraude (1)"]).to_string())
print(f"Precisión: {precision_score(y_test, predicciones, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predicciones, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, predicciones, zero_division=0):.4f}")
tn, fp, fn, tp = matriz.ravel()
print(f"Verdaderos negativos: {tn} | Falsos positivos: {fp}")
print(f"Falsos negativos: {fn} | Verdaderos positivos: {tp}")
