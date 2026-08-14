import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from scoring import load_artifact, score_pair

app = FastAPI()

EXPECTED_API_KEY = os.getenv("ML_SERVICE_API_KEY")

# Loaded once at process startup, not per request.
MODEL_ARTIFACT = load_artifact()


def require_api_key(x_api_key: str = Header(...)):
    if not EXPECTED_API_KEY:
        raise HTTPException(status_code=500, detail="ML_SERVICE_API_KEY is not set on the server")
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ScoreRequest(BaseModel):
    resume: str
    jd: str


class ScoreResponse(BaseModel):
    score: float
    label: str
    features: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score(body: ScoreRequest):
    return score_pair(body.resume, body.jd, MODEL_ARTIFACT)
