from fastapi import FastAPI, Request
import random
import uuid

app = FastAPI()

current_email = {"text": "", "label": ""}

EMAILS = [
    {"text": "Congratulations! You won a gift card. Click here!", "label": "SPAM"},
    {"text": "Hi team, please review the attached Q3 report.", "label": "WORK"},
    {"text": "Hey, can you pick up milk on your way home?", "label": "PERSONAL"},
    {"text": "URGENT: Your account has been compromised.", "label": "SPAM"},
    {"text": "Are we still meeting for lunch tomorrow?", "label": "PERSONAL"},
    {"text": "Server deployment tonight at 2 AM. Please merge.", "label": "WORK"},
    {"text": "FREE cruise selected for you! Limited slots!", "label": "SPAM"},
    {"text": "Can you review my PR before EOD?", "label": "WORK"},
    {"text": "Moms birthday next week, dont forget to call!", "label": "PERSONAL"},
]

@app.head("/health")
@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    global current_email
    current_email = random.choice(EMAILS)
    return {
        "observation": {
            "email_text": current_email["text"],
            "done": False,
            "reward": 0.0
        },
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
async def step(request: Request):
    global current_email
    body = await request.json()
    category = body.get("category", "").upper().strip()
    reward = 1.0 if category == current_email["label"] else 0.0
    return {
        "observation": {
            "email_text": current_email["text"],
            "done": True,
            "reward": reward
        },
        "reward": reward,
        "done": True
    }

@app.get("/state")
def state():
    return {"episode_id": str(uuid.uuid4()), "step_count": 0}
