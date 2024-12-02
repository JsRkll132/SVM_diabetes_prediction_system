import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Cargar el conjunto de datos de diabetes
diabetes_dataset = pd.read_csv('./content/diabetes.csv')

# Mostrar las primeras filas del conjunto de datos
print(diabetes_dataset.head())

# Separar los datos y las etiquetas
X = diabetes_dataset.drop(columns='Outcome', axis=1).values
Y = diabetes_dataset['Outcome'].values

# Convertir las etiquetas 0, 1 en -1, 1 (como requiere el SVM)
Y = 2 * Y - 1

# Verificar si hay desbalance en las clases
class_counts = Counter(Y)
print(f"Distribución de clases antes de SMOTE: {class_counts}")

# Aplicar SMOTE si hay desbalance significativo
if min(class_counts.values()) / max(class_counts.values()) < 0.8:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(X, Y)
    class_counts = Counter(Y)
    print(f"Distribución de clases después de SMOTE: {class_counts}")

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# 1. Implementar el Kernel RBF (Radial Basis Function)
def rbf_kernel(X, X2, gamma=1.0):
    """
    Calcula la matriz de kernel RBF entre dos matrices X y X2.
    """
    pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X, X2.T)
    return np.exp(-gamma * pairwise_sq_dists)

# 2. Función de optimización usando el método de los multiplicadores de Lagrange
def train_svm(X, Y, C=10.0, gamma=1.0):
    """
    Entrena el modelo SVM con Kernel RBF usando el método de los multiplicadores de Lagrange (sin librerías).
    """
    n_samples, n_features = X.shape
    # Crear la matriz del kernel RBF
    K = rbf_kernel(X, X, gamma)
    
    # Matriz de coeficientes de la optimización
    H = np.outer(Y, Y) * K
    P = H
    
    # Vector b y restricciones
    q = -np.ones(n_samples)
    G = -np.eye(n_samples)
    h = np.zeros(n_samples)
    
    # Usar un optimizador para resolver el problema cuadrático
    from scipy.optimize import minimize
    result = minimize(lambda alpha: 0.5 * np.dot(alpha.T, np.dot(P, alpha)) - np.sum(alpha),
                      np.zeros(n_samples), constraints={'type': 'eq', 'fun': lambda alpha: np.sum(alpha * Y)},
                      bounds=[(0, C) for _ in range(n_samples)])
    
    # Los multiplicadores de Lagrange
    alpha = result.x
    
    # Encontrar el sesgo b (usando los puntos de soporte)
    support_vectors = np.where(alpha > 1e-5)[0]
    b = np.mean(Y[support_vectors] - np.dot(K[support_vectors], alpha * Y))
    
    return alpha, b, K

# 3. Predecir usando el modelo SVM con Kernel RBF
def predict(X, alpha, Y, X_train, b, gamma=1.0):
    """
    Realiza la predicción usando el modelo entrenado.
    """
    K = rbf_kernel(X_train, X, gamma)
    return np.sign(np.dot(K.T, alpha * Y) + b)

# 4. Entrenar el modelo
alpha, b, K_train = train_svm(X_train, Y_train, C=1.0, gamma=0.01)

# 5. Evaluar el modelo en el conjunto de entrenamiento
train_predictions = predict(X_train, alpha, Y_train, X_train, b, gamma=0.01)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Accuracy on training data (RBF kernel): {train_accuracy * 100:.2f}%")

# Evaluar el modelo en el conjunto de prueba
test_predictions = predict(X_test, alpha, Y_train, X_train, b, gamma=0.01)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Accuracy on test data (RBF kernel): {test_accuracy * 100:.2f}%")

# 6. Realizar una predicción para una nueva entrada de datos
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convertir los datos de entrada a un array de numpy
input_data_as_numpy_array = np.asarray(input_data)

# Cambiar la forma de los datos para predecir una sola instancia
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Estandarizar los datos de entrada
std_data = scaler.transform(input_data_reshaped)

# Realizar la predicción con el modelo SVM RBF
prediction = predict(std_data, alpha, Y_train, X_train, b, gamma=0.01)
print("Prediction (RBF Kernel):", prediction)

# Imprimir el resultado de la predicción
if prediction[0] == 1:
    print("The person is diabetic (RBF Kernel)")
else:
    print("The person is not diabetic (RBF Kernel)")


# Guardar el modelo y el scaler
model_data = {
    "alpha": alpha,
    "b": b,
    "X_train": X_train,
    "Y_train": Y_train,
    "scaler": scaler
}
with open("svm_model.pkl", "wb") as file:
    pickle.dump(model_data, file)
print("Modelo guardado exitosamente.")
