import random
import uuid
from fastapi import FastAPI, Request
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

class ActionRequest(BaseModel):
    category: str

@app.get("/")
def root():
    return {"status": "ok", "service": "email-triage-env"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset():
    global current_email
    current_email = random.choice(EMAILS)
    return {"observation": {"email_text": current_email["text"], "done": False, "reward": 0.0}, "reward": 0.0, "done": False}

@app.post("/step")
def step(action: ActionRequest):
    guess = action.category.upper().strip()
    reward = 1.0 if guess == current_email["label"] else 0.0
    return {"observation": {"email_text": current_email["text"], "done": True, "reward": reward}, "reward": reward, "done": True}

@app.get("/state")
def state():
    return {"episode_id": str(uuid.uuid4()), "step_count": 0}
