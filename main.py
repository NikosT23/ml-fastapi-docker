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
        <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    </head>
    <body class="bg-gray-900 text-white flex items-center justify-center min-h-screen">
        <div class="bg-gray-800 p-8 rounded-xl shadow-2xl w-full max-w-md border border-gray-700">
            <h1 class="text-2xl font-bold mb-2 text-center text-indigo-400">Iris Species Classifier</h1>
            <p class="text-gray-400 text-sm mb-6 text-center">Adjust the flower measurements below to predict its species.</p>
            
            <form id="prediction-form" class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300">Sepal Length (cm)</label>
                    <input type="number" step="0.1" id="sepal_length" value="5.1" class="w-full mt-1 p-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300">Sepal Width (cm)</label>
                    <input type="number" step="0.1" id="sepal_width" value="3.5" class="w-full mt-1 p-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300">Petal Length (cm)</label>
                    <input type="number" step="0.1" id="petal_length" value="1.4" class="w-full mt-1 p-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300">Petal Width (cm)</label>
                    <input type="number" step="0.1" id="petal_width" value="0.2" class="w-full mt-1 p-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none">
                </div>
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2 px-4 rounded-lg transition duration-200">Predict Species</button>
            </form>

            <div id="result-container" class="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700 text-center hidden">
                <span class="text-gray-400 text-sm">Predicted Species:</span>
                <div id="prediction-text" class="text-xl font-bold text-emerald-400 mt-1"></div>
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

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const container = document.getElementById('result-container');
                const text = document.getElementById('prediction-text');
                
                container.classList.remove('hidden');
                text.textContent = result.prediction.toUpperCase();
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
    
    return {"predicted_species": predicted_species}

    
