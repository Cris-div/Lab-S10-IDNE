import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=650, n_features=8, n_informative=5, n_redundant=1,
    n_classes=2, weights=[0.65, 0.35], random_state=29,
)
datos = pd.DataFrame(X, columns=[f"variable_{i}" for i in range(1, 9)])
datos["cancelacion"] = y
# Se estratifica para conservar aproximadamente las proporciones de clase.
# Con 650 registros, sklearn asigna 487 a entrenamiento y 163 a prueba.
X_train, X_test, y_train, y_test = train_test_split(
    datos.drop(columns="cancelacion"), datos["cancelacion"],
    test_size=0.25, random_state=29, stratify=datos["cancelacion"],
)
# Escalar las variables antes de entrenar el modelo.
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = SVC(kernel="rbf", C=1, gamma="scale")
modelo.fit(X_train_escalado, y_train)
predicciones = modelo.predict(X_test_escalado)

precision = precision_score(y_test, predicciones, pos_label=1, zero_division=0)
recall = recall_score(y_test, predicciones, pos_label=1, zero_division=0)
f1 = f1_score(y_test, predicciones, pos_label=1, zero_division=0)
matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])
tn, fp, fn, tp = matriz.ravel()

print("EJERCICIO 4 - CANCELACIÓN DE SUSCRIPCIONES")
print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
print("\na) Evaluación sobre datos de prueba (clase positiva = 1)")
print(f"Precisión: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Matriz de confusión (filas = real; columnas = predicho):")
print(pd.DataFrame(matriz, index=["No cancela (0)", "Cancela (1)"],
                   columns=["No cancela (0)", "Cancela (1)"]).to_string())
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

falsos_negativos = X_test.loc[(y_test == 1) & (predicciones == 0)].copy()
falsos_negativos["real"] = 1
falsos_negativos["predicho"] = 0
print(f"\nFalsos negativos: {fn} usuarios que cancelan y no fueron detectados.")
print("El índice identifica la fila original, comenzando en cero.")
print(falsos_negativos.to_string(float_format=lambda x: f"{x:.4f}"))
print("Estos usuarios podrían quedar fuera de las acciones de retención.")

cumple_recall = recall >= 0.75
cumple_precision = precision >= 0.60
print("\nb) Verificación de umbrales")
print(f"Recall >= 0.75: {'cumple' if cumple_recall else 'no cumple'}")
print(f"Precisión >= 0.60: {'cumple' if cumple_precision else 'no cumple'}")
decision = ("aprobar preliminarmente" if cumple_recall and cumple_precision
            else "ajustar el modelo")
print(f"Decisión: {decision}")
print("Advertencia: antes del despliegue se requiere una revisión adicional")
print("de sesgos, fuga de información y utilidad del negocio. Cumplir los")
print("umbrales con datos simulados no garantiza el desempeño en usuarios reales.")
