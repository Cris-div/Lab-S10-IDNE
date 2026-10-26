import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Datos del laboratorio. 0 = no defectuoso; 1 = defectuoso.
datos = pd.DataFrame({
    "temperatura": [60, 62, 65, 67, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100, 103, 105, 108],
    "presion": [10, 11, 12, 13, 12, 14, 15, 15, 16, 17, 19, 20, 22, 23, 25, 26, 28, 30, 32, 35],
    "vibracion": [2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20],
    "tiempo_produccion": [30, 32, 35, 36, 38, 40, 43, 45, 47, 50, 55, 58, 60, 63, 66, 70, 75, 78, 80, 85],
    "defectuoso": [0] * 10 + [1] * 10,
})
X = datos.drop(columns="defectuoso")
y = datos["defectuoso"]
# Los tres modelos utilizan exactamente la misma división para compararlos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=29, stratify=y
)
# Estandarizar usando solo las estadísticas del entrenamiento.
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

resultados = []
for kernel in ["linear", "rbf", "poly"]:
    modelo = SVC(kernel=kernel, C=1, gamma="scale", degree=3)
    modelo.fit(X_train_escalado, y_train)
    predicciones = modelo.predict(X_test_escalado)
    resultados.append({
        "Kernel": kernel,
        "Accuracy": accuracy_score(y_test, predicciones),
        "F1-score": f1_score(y_test, predicciones, zero_division=0),
    })

print("EJERCICIO 3 - COMPARACIÓN DE KERNELS")
print(pd.DataFrame(resultados).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("La prueba contiene solo cinco productos; se necesitan más datos para")
print("confirmar la capacidad de generalización y seleccionar un kernel.")
