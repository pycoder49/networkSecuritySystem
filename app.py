from network_security.logging.logger import logging
from network_security.exceptions.exception import NetworkSecurityException
from network_security.pipeline.training_pipeline import TrainingPipeline
from network_security.utils.main_utils.utils import load_object
from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.constants.training_pipeline import (
    DATA_INGESTION_DATABASE_NAME, 
    DATA_INGESTION_COLLECTION_NAME
)

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from starlette.responses import RedirectResponse


from dotenv import load_dotenv

import pymongo
import pandas as pd

import os, sys
import certifi


ca = certifi.where()
load_dotenv()
mongodb_url = os.getenv("MONGODB_URI")
print("MongoDB URL loaded with certifi:", mongodb_url, ca)

# connecting to mongodb
client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# defining the app
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# jinja2 templates
templates = Jinja2Templates(directory="templates")

# defining methods
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response(content="Training was successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # reading the incoming file
        df = pd.read_csv(file.file)

        # getting the preprocessor and model
        preprocessor = load_object("final_model/transformer.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)

        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)

        df["predicted_column"] = y_pred
        print(df["predicted_column"])

        # outputting to file
        df.to_csv("prediction_output/output.csv", index=False)

        # adding to templates
        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)

    

if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)
