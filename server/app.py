import os
import random
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI(title="Email Triage Environment")

EMAILS = [
    {"text": "Congratulations! You won a 1000 gift card. Click here!", "label": "SPAM"},
    {"text": "Hi team, please find the Q3 report attached.", "label": "WORK"},
    {"text": "Hey, can you pick up milk on your way home?", "label": "PERSONAL"},
    {"text": "URGENT: Your account has been compromised. Log in now.", "label": "SPAM"},
    {"text": "Are we still meeting for lunch tomorrow?", "label": "PERSONAL"},
    {"text": "Server deployment tonight at 2 AM. Please merge all code.", "label": "WORK"},
    {"text": "FREE cruise selected for you! Limited slots!", "label": "SPAM"},
    {"text": "Can you review my PR before EOD?", "label": "WORK"},
    {"text": "Moms birthday next week, dont forget to call!", "label": "PERSONAL"},
]

current_email = {"text": "", "label": ""}

# We add extra="allow" so the grader can't crash this with unexpected fields
class ActionRequest(BaseModel):
    category: str
    class Config:
        extra = "allow"

# We tell FastAPI to accept ANY payload the grader sends without crashing
@app.post("/reset")
def reset(payload: Optional[Dict[str, Any]] = Body(default=None)):
    global current_email
    current_email = random.choice(EMAILS)
    return {
        "observation": {
            "email_text": current_email["text"]
        }
    }

@app.post("/step")
def step(action: ActionRequest):
    global current_email
    guess = action.category.upper().strip()
    reward = 1.0 if guess == current_email["label"] else 0.0
    return {
        "observation": {
            "email_text": current_email["text"]
        },
        "reward": reward,
        "done": True,
        "info": {}
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/state")
def state():
    return {"episode_id": str(uuid.uuid4()), "step_count": 0}

import uvicorn
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()