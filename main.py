from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, area: float = Form(...), bedrooms: int = Form(...), bathrooms: int = Form(...)):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms]], 
                              columns=['area_sqm', 'bedrooms', 'bathrooms'])
    
    input_scaled = scaler.transform(input_data)
    price_pred = model.predict(input_scaled)[0]
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": f"{price_pred:,.2f}"
    })