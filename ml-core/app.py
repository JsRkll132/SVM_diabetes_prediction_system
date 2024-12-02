from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS
# Cargar el modelo
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

# Crear la aplicación Flask
app = Flask(__name__)

CORS(app=app,origins="*")
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Obtener los datos JSON enviados por el cliente
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extraer los datos de entrada
        input_data = [
            data.get("Pregnancies"),
            data.get("Glucose"),
            data.get("BloodPressure"),
            data.get("SkinThickness"),
            data.get("Insulin"),
            data.get("BMI"),
            data.get("DiabetesPedigreeFunction"),
            data.get("Age")
        ]
        
        # Verificar que no haya valores faltantes
        if None in input_data:
            return jsonify({'error': 'Some input fields are missing'}), 400
        print(input_data)
        # Convertir los datos de entrada en un array de numpy
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Estandarizar los datos
        std_data = loaded_scaler.transform(input_data_as_numpy_array)
        
        # Realizar la predicción
        prediction = predict(std_data, loaded_alpha, loaded_Y_train, loaded_X_train, loaded_b, gamma=0.01)
        
        # Retornar el resultado
        result = "diabetic" if prediction[0] == 1 else "not diabetic"
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
