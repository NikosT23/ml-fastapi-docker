from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Iris Model API")

model = joblib.load("iris_model.joblib")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

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

    
