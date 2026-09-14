# ============================================================================
# HURUDZA NZWISISO - FASTAPI BACKEND
# Run with: uvicorn app:app --reload --port 8000
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os

from hurudza import HurudzaModel, HurudzaNLSEngine

# -------- Load the trained model at startup --------
MODEL_PATH = os.getenv("HURUDZA_MODEL_PATH", "hurudza_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file '{MODEL_PATH}' not found. "
        "Run `python hurudza.py` first to train and save the model."
    )

model = HurudzaModel.load(MODEL_PATH)
engine = HurudzaNLSEngine(model)

# -------- FastAPI app --------
app = FastAPI(
    title="Hurudza Nzwisiso API",
    description="Agricultural advisory NLP system for crop health, pest, disease, soil, and water management.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Request schemas --------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=2, description="User's agricultural query")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=2)

class QARequest(BaseModel):
    question: str = Field(..., min_length=2)
    context: Optional[str] = Field(None, description="Optional context for extractive QA")

class SummarizeRequest(BaseModel):
    text: str
    max_sentences: int = 3


# -------- Endpoints --------
@app.get("/")
def root():
    return {
        "service": "Hurudza Nzwisiso",
        "version": "1.0.0",
        "status": "online",
        "model_trained_at": model.trained_at,
        "categories": model.categories,
        "endpoints": [
            "/analyze", "/predict", "/entities", "/qa",
            "/summarize", "/urgency", "/health", "/categories"
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model.is_trained}


@app.get("/categories")
def categories():
    return {
        "categories": model.categories,
        "metrics": {
            "accuracy": model.training_metrics.get('accuracy'),
            "f1_weighted": model.training_metrics.get('f1_weighted'),
            "num_train": model.training_metrics.get('num_train'),
        }
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        return engine.analyze(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return model.predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/entities")
def entities(req: PredictRequest):
    try:
        return engine.extract_entities(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qa")
def qa(req: QARequest):
    try:
        return engine.answer_question(req.question, req.context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    try:
        return {"summary": engine.summarize(req.text, req.max_sentences)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/urgency")
def urgency(req: PredictRequest):
    try:
        return engine.analyze_urgency(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))