
import numpy as np
import pickle
#from train_model import predict
# Cargar el modelo y realizar predicciones
with open("svm_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Cargar los parámetros del modelo
loaded_alpha = loaded_model["alpha"]
loaded_b = loaded_model["b"]
loaded_X_train = loaded_model["X_train"]
loaded_Y_train = loaded_model["Y_train"]
loaded_scaler = loaded_model["scaler"]
def rbf_kernel(X, X2, gamma=1.0):
    """
    Calcula la matriz de kernel RBF entre dos matrices X y X2.
    """
    pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X, X2.T)
    return np.exp(-gamma * pairwise_sq_dists)

def predict(X, alpha, Y, X_train, b, gamma=1.0):
    """
    Realiza la predicción usando el modelo entrenado.
    """
    K = rbf_kernel(X_train, X, gamma)
    return np.sign(np.dot(K.T, alpha * Y) + b)

# Ejemplo de predicción con el modelo cargado
input_data = (5, 132, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = loaded_scaler.transform(input_data_as_numpy_array)
prediction = predict(std_data, loaded_alpha, loaded_Y_train, loaded_X_train, loaded_b, gamma=0.01)

# Imprimir el resultado
if prediction[0] == 1:
    print("The person is diabetic (RBF Kernel, loaded model)")
else:
    print("The person is not diabetic (RBF Kernel, loaded model)")