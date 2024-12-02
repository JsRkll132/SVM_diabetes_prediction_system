import React, { useState } from "react";
import axios from "axios";
import Swal from "sweetalert2";
import "sweetalert2/dist/sweetalert2.min.css";

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
      const numericData = Object.keys(formData).reduce((acc, key) => {
        acc[key] = parseFloat(formData[key]);
        return acc;
      }, {});

      const response = await axios.post("http://localhost:5000/predict", numericData);

      const prediction = response.data.prediction;

      Swal.fire({
        title: "Resultado de la Predicción",
        text:
          prediction === "diabetic"
            ? "El paciente es diabético."
            : "El paciente no es diabético.",
        icon: prediction === "diabetic" ? "warning" : "info",
        confirmButtonText: "Aceptar",
        confirmButtonClass: "btn btn-success",
        buttonsStyling: true, // Habilita estilos personalizados con Bootstrap
        customClass: {
          confirmButton: prediction === "diabetic" ? "btn btn-danger" : "btn btn-success",
          popup: "border border-secondary rounded-3 shadow", // Estilo de borde Bootstrap
        },
      });

      setError(null);
    } catch (err) {
      setError(err.response?.data?.error || "Ocurrió un error");
    }
  };

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

  const inputRanges = {
    Pregnancies: { min: 0, max: 17 },
    Glucose: { min: 65, max: 500 },
    BloodPressure: { min: 0, max: 122 },
    SkinThickness: { min: 0, max: 99 },
    Insulin: { min: 0, max: 846 },
    BMI: { min: 14, max: 67.1 },
    DiabetesPedigreeFunction: { min: 0.078, max: 2.420, step: 0.01 },
    Age: { min: 21, max: 81 },
  };

  return (
    <div className="container py-4">
      <h1 className="text-center mb-4">Predicción de Diabetes</h1>
      <form onSubmit={handleSubmit} className="needs-validation">
        <div className="row g-3 justify-content-center">
          {Object.keys(formData).map((field) => (
            <div key={field} className="col-md-6">
              <label className="form-label d-flex justify-content-between">
                <span>{fieldLabels[field]}:</span>
                {field !== "Pregnancies" && (
                  <span className="text-muted">
                    <strong>
                      ({inputRanges[field].min} - {inputRanges[field].max})
                    </strong>
                  </span>
                )}
              </label>
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

      {error && (
        <div className="alert alert-danger mt-4 text-center">
          <h4>Error: {error}</h4>
        </div>
      )}
    </div>
  );
};

export default PredictPage;
