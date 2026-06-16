import os

from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()

EXPECTED_API_KEY = os.getenv("ML_SERVICE_API_KEY")


def require_api_key(x_api_key: str = Header(...)):
    if not EXPECTED_API_KEY:
        raise HTTPException(status_code=500, detail="ML_SERVICE_API_KEY is not set on the server")
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/protected-example", dependencies=[Depends(require_api_key)])
def protected_example():
    return {"message": "you have a valid api key"}
