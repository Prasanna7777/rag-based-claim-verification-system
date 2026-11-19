import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from collections import defaultdict
from pipeline import ClaimVerificationPipeline

app = FastAPI(title="Claim Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prasanna7777.github.io", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing pipeline...")
pipeline = ClaimVerificationPipeline()
print("Pipeline ready.")

# --- simple in-memory rate limiting (per IP, sliding window) ---
RATE_LIMIT = 10          # max requests
RATE_WINDOW = 60         # per this many seconds
request_log = defaultdict(list)


def check_rate_limit(client_ip: str):
    now = time.time()
    request_log[client_ip] = [t for t in request_log[client_ip] if now - t < RATE_WINDOW]
    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment and try again.")
    request_log[client_ip].append(now)


class VerifyRequest(BaseModel):
    text: str
    top_k: int = 3

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Claim text cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Claim text is too long (max 2000 characters).")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("top_k must be between 1 and 10.")
        return v


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong while processing your request. Please try again."},
    )


@app.post("/verify")
def verify(req: VerifyRequest, request: Request):
    check_rate_limit(request.client.host)

    try:
        results = pipeline.verify(req.text, top_k=req.top_k)
    except Exception as e:
        print(f"Pipeline error for input '{req.text}': {e}")
        raise HTTPException(
            status_code=502,
            detail="Could not complete verification — an upstream service (search or generation) may be temporarily unavailable.",
        )

    return {"submission": req.text, "results": results}


@app.get("/")
def root():
    return {"status": "Claim Verification API is running"}