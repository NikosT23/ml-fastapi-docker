from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import HTMLResponse

app = FastAPI(title="Iris Model API")

model = joblib.load("iris_model.joblib")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris ML Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #0f172a;
                color: #f8fafc;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .card {
                background-color: #1e293b;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                width: 350px;
                border: 1px solid #334155;
            }
            h1 { font-size: 22px; margin-bottom: 5px; color: #818cf8; text-align: center; }
            p { font-size: 13px; color: #94a3b8; text-align: center; margin-bottom: 20px; }
            .input-group { margin-bottom: 15px; }
            label { display: block; font-size: 12px; margin-bottom: 5px; color: #cbd5e1; }
            input {
                width: 100%;
                padding: 8px;
                background-color: #0f172a;
                border: 1px solid #475569;
                border-radius: 6px;
                color: white;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                padding: 10px;
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover { background-color: #4f46e5; }
            #result-container {
                margin-top: 20px;
                padding: 12px;
                background-color: #0f172a;
                border-radius: 6px;
                text-align: center;
                border: 1px solid #334155;
                display: none;
            }
            #prediction-text { font-size: 20px; font-weight: bold; color: #34d399; margin-top: 5px; text-transform: uppercase; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Iris Species Classifier</h1>
            <p>Adjust the measurements to predict the flower type.</p>
            
            <form id="prediction-form">
                <div class="input-group">
                    <label>Sepal Length (cm)</label>
                    <input type="number" step="0.1" id="sepal_length" value="5.1">
                </div>
                <div class="input-group">
                    <label>Sepal Width (cm)</label>
                    <input type="number" step="0.1" id="sepal_width" value="3.5">
                </div>
                <div class="input-group">
                    <label>Petal Length (cm)</label>
                    <input type="number" step="0.1" id="petal_length" value="1.4">
                </div>
                <div class="input-group">
                    <label>Petal Width (cm)</label>
                    <input type="number" step="0.1" id="petal_width" value="0.2">
                </div>
                <button type="submit">Predict Species</button>
            </form>

            <div id="result-container">
                <span style="font-size: 12px; color: #94a3b8;">Predicted Species:</span>
                <div id="prediction-text"></div>
            </div>
        </div>

        <script>
            document.getElementById('prediction-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const data = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();
                    const container = document.getElementById('result-container');
                    const text = document.getElementById('prediction-text');
                    
                    container.style.display = 'block';
                    text.textContent = result.prediction;
                } catch (error) {
                    alert('Error connecting to prediction API');
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict_species(data: IrisData):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    prediction = model.predict(features)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = species_map[int(prediction[0])]
    
    return {"prediction": predicted_species}

    
