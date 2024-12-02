import React, { useState } from "react";
import axios from "axios";

const PredictPage = () => {
  const [formData, setFormData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Convertir los valores a números
      const numericData = Object.keys(formData).reduce((acc, key) => {
        acc[key] = parseFloat(formData[key]);
        return acc;
      }, {});

      const response = await axios.post("http://localhost:5000/predict", numericData);
      setPrediction(response.data.prediction);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.error || "Ocurrió un error");
      setPrediction(null);
    }
  };

  // Traducción de las etiquetas para el frontend
  const fieldLabels = {
    Pregnancies: "Número de embarazos",
    Glucose: "Glucosa en sangre (mg/dL)",
    BloodPressure: "Presión arterial (mmHg)",
    SkinThickness: "Grosor del pliegue cutáneo (mm)",
    Insulin: "Insulina sérica (µU/mL)",
    BMI: "Índice de masa corporal (kg/m²)",
    DiabetesPedigreeFunction: "Predisposición genética",
    Age: "Edad (años)",
  };

  // Rango de valores válidos según la tabla
  const inputRanges = {
    Pregnancies: { min: 0, max: 17 },
    Glucose: { min: 0, max: 500 },
    BloodPressure: { min: 0, max: 122 },
    SkinThickness: { min: 0, max: 99 },
    Insulin: { min: 0, max: 846 },
    BMI: { min: 14, max: 67.1 },
    DiabetesPedigreeFunction: { min: 0.078, max: 2.42, step: 0.01 },
    Age: { min: 21, max: 81 },
  };

  return (
    <div className="container py-4">
      <h1 className="text-center mb-4">Predicción de Diabetes</h1>
      <form onSubmit={handleSubmit} className="needs-validation">
        <div className="row g-3 justify-content-center">
          {Object.keys(formData).map((field) => (
            <div key={field} className="col-md-6">
              <label className="form-label">{fieldLabels[field]}:</label>
              <input
                type="number"
                step={inputRanges[field].step || "any"}
                name={field}
                value={formData[field]}
                onChange={handleChange}
                required
                min={inputRanges[field].min}
                max={inputRanges[field].max}
                className="form-control"
              />
            </div>
          ))}
        </div>
        <div className="text-center mt-4">
          <button className="btn btn-primary btn-lg" type="submit">
            Predecir
          </button>
        </div>
      </form>

      {prediction && (
        <div className="alert alert-success mt-4 text-center">
          <h4>Predicción: {prediction}</h4>
        </div>
      )}
      {error && (
        <div className="alert alert-danger mt-4 text-center">
          <h4>Error: {error}</h4>
        </div>
      )}
    </div>
  );
};

export default PredictPage;
