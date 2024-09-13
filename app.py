# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the app
app = FastAPI()

# Load the trained model and model columns
model = joblib.load("car_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Define the input data model
class CarFeatures(BaseModel):
    buying: str
    maint: str
    doors: str
    persons: str
    lug_boot: str
    safety: str

# Create a prediction endpoint
@app.post("/predict")
def predict(features: CarFeatures):
    # Convert the input into a dataframe
    input_data = pd.DataFrame([features.dict()])
    input_data = pd.get_dummies(input_data)
    
    # Reindex the input data to match model columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the prediction
    return {"prediction": prediction[0]}
