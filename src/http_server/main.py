from fastapi import FastAPI
import joblib
import logging
import numpy as np
import os
from pydantic import BaseModel

from mlonactions.model import Model

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

class SingleInferenceInput(BaseModel):
    x: list
class SingleInferencePrediction(BaseModel):
    y: int

model_path = "../data/model/model.joblib"
app = FastAPI()
model = joblib.load(model_path)

@app.post("/single_predict/")
def single_inference(item: SingleInferenceInput)->SingleInferencePrediction:
    
    x = np.asarray(item.x)

    logging.info(f"single_inference called with {x.shape=}")
    assert len(x.shape) == 1
    X = np.expand_dims(x, axis=0)
    return {"y": model.predict(X).tolist()[0]}