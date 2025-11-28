# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from PlantInferencePipeline import PlantInferencePipeline



app = FastAPI(title="Plant Care API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "./Model/best-plant-types-model-v12.ckpt")

pipeline = PlantInferencePipeline(
    model_path=MODEL_PATH,
    api_key=API_KEY
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        result = pipeline.inference(file.file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
