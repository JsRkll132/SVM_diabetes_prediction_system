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
      setError(err.response?.data?.error || "An error occurred");
      setPrediction(null);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Diabetes Prediction</h1>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map((field) => (
          <div key={field} style={{ marginBottom: "15px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>
              {field}:
            </label>
            <input
              type="number"
              step="any"
              name={field}
              value={formData[field]}
              onChange={handleChange}
              required
              style={{
                padding: "10px",
                width: "100%",
                boxSizing: "border-box",
              }}
            />
          </div>
        ))}
        <button  className="btn btn-primary" type="submit" style={{ padding: "10px 20px", cursor: "pointer" }}>
          Predict
        </button>
      </form>

      {prediction && (
        <div style={{ marginTop: "20px", color: "green" }}>
          <h3>Prediction: {prediction}</h3>
        </div>
      )}
      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          <h3>Error: {error}</h3>
        </div>
      )}
    </div>
  );
};

export default PredictPage;
