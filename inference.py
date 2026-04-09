import os
import sys
from openai import OpenAI
import requests

SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
MAX_STEPS = 5

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = "You are an Email Triage Assistant. Classify the email into exactly one of: [SPAM] [WORK] [PERSONAL]. Respond ONLY with the category in brackets."

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

def reset_env(task):
    try:
        r = requests.post(f"{SERVER_URL}/reset", json={"task": task}, timeout=20)
        return r.json()
    except Exception as e:
        print(f"[DEBUG] reset failed: {e}", flush=True)
        return {"observation": {"email_text": "Test email"}}

def step_env(category):
    try:
        r = requests.post(f"{SERVER_URL}/step", json={"category": category}, timeout=20)
        return r.json()
    except Exception as e:
        print(f"[DEBUG] step failed: {e}", flush=True)
        return {"reward": 0.0, "done": True}

def get_llm_action(client, email_text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text}
        ],
        max_tokens=10,
        temperature=0.0,
    )
    output = response.choices[0].message.content.strip().upper()
    print(f"[DEBUG] LLM raw output: {output}", flush=True)
    if "SPAM" in output:
        return "SPAM"
    elif "WORK" in output:
        return "WORK"
    elif "PERSONAL" in output:
        return "PERSONAL"
    else:
        return "SPAM"

def run_task(client, task_name):
    all_rewards = []
    total_score = 0.0
    step_count = 0

    try:
        obs_data = reset_env(task_name)
        email_text = obs_data.get("observation", {}).get("email_text", "Test email")

        print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}", flush=True)

        for step in range(1, MAX_STEPS + 1):
            action = get_llm_action(client, email_text)
            result = step_env(action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", True))

            all_rewards.append(reward)
            step_count = step
            total_score = reward

            print(f"[STEP] step={step} action=[{action}] reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            if done:
                break

        success = total_score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} score={total_score:.2f} rewards={rewards_str}", flush=True)

    except Exception as e:
        print(f"[DEBUG] {e}", flush=True)
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
        raise

    return total_score

if __name__ == "__main__":
    print("=" * 50, flush=True)
    print("Email Triage - Inference Script", flush=True)
    print("=" * 50, flush=True)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] API_KEY={'set' if API_KEY else 'NOT SET'}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = []
    for task in TASKS:
        score = run_task(client, task)
        scores.append(score)
        print(flush=True)

    avg = sum(scores) / len(scores)
    print(f"Average score: {avg:.2f}", flush=True)
    sys.exit(0)
