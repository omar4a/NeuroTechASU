from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .speller import API

app = FastAPI(
    title="Speller API",
    description="Predictive speller logic endpoint.",
)

# Allow webpages to hit this endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

speller_api = API()

class PredictRequest(BaseModel):
    prefix: str = ""
    sentence: str = ""
    context: str = ""

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict next words given optional prefix, sentence, and context.
    """
    try:
        words = speller_api.predict_words(
            prefix=request.prefix,
            sentence=request.sentence,
            context=request.context,
        )
        return {"predictions": words}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
