from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
RKLLM_API_URL = 'http://192.168.1.157:1306/generate'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your specific origin(s)
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Reuse your LLM proxy function
async def generate_response(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            RKLLM_API_URL,
            json={"prompt": prompt},
            timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()
        # Expecting {'response': '...'}
        return data.get("response", "")

# Define the shape of the incoming POST body
class PromptRequest(BaseModel):
    prompt: str

# New HTTP POST endpoint
@app.post("/generate")
async def generate_endpoint(req: PromptRequest):
    try:
        bot_text = await generate_response(req.prompt)
        return {"response": bot_text}
    except httpx.HTTPStatusError as e:
        # Forward 4xx/5xx from your rkLLM service
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        # Catch-all for other errors
        raise HTTPException(status_code=500, detail=str(e))
