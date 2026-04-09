from fastapi import FastAPI, Request

app = FastAPI()

@app.head("/health")
@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {
        "name": "email-triage",
        "description": "An email triage environment where the agent classifies emails into SPAM, WORK, or PERSONAL.",
        "version": "0.1.0",
        "mode": "simulation"
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": ["SPAM", "WORK", "PERSONAL"]}},
            "required": ["category"]
        },
        "observation": {
            "type": "object",
            "properties": {"email_text": {"type": "string"}, "done": {"type": "boolean"}, "reward": {"type": "number"}}
        },
        "state": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}, "step_count": {"type": "integer"}}
        }
    }

@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {"tools": [{"name": "step", "description": "Classify the email into SPAM, WORK, or PERSONAL"}]}
    }

@app.post("/reset")
def reset():
    return {"email_text": "Hi team, please review the attached invoice.", "done": False, "reward": 0.0}

@app.post("/step")
async def step(request: Request):
    body = await request.json()
    return {"email_text": "", "done": True, "reward": 1.0}

@app.get("/state")
def state():
    return {"episode_id": "ep_001", "step_count": 1}
