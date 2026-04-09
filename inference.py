import os
import sys
import requests
from openai import OpenAI

SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
MAX_STEPS = 5

SYSTEM_PROMPT = "You are an Email Triage Assistant. Classify the email into exactly one of: [SPAM] [WORK] [PERSONAL]. Respond ONLY with the category in brackets."

TASKS = ["easy_triage", "medium_triage", "hard_triage"]

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def reset_env(task):
    try:
        r = requests.post(f"{SERVER_URL}/reset", json={"task": task}, timeout=20)
        return r.json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}", file=sys.stderr)
        return {"observation": {"email_text": "Test email"}}


def step_env(category):
    try:
        r = requests.post(f"{SERVER_URL}/step", json={"category": category}, timeout=20)
        return r.json()
    except Exception as e:
        print(f"[ERROR] step failed: {e}", file=sys.stderr)
        return {"reward": 0.0, "done": True}


def get_llm_action(email_text):
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )
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
    print(f"[DEBUG] LLM raw output: {output}", file=sys.stderr)
    if "SPAM" in output:
        return "SPAM"
    elif "WORK" in output:
        return "WORK"
    elif "PERSONAL" in output:
        return "PERSONAL"
    else:
        print(f"[WARN] Unexpected LLM output: {output}, defaulting to SPAM", file=sys.stderr)
        return "SPAM"


def run_task(task_name):
    all_rewards = []
    total_score = 0.0
    step_count = 0
    try:
        obs_data = reset_env(task_name)
        email_text = obs_data.get("observation", {}).get("email_text", "Test email")
        print(f"[START] task={task_name} env=email-triage")
        for step in range(1, MAX_STEPS + 1):
            action = get_llm_action(email_text)
            result = step_env(action)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", True))
            all_rewards.append(reward)
            step_count = step
            total_score = reward
            print(f"[STEP] step={step} action=[{action}] reward={reward:.2f} done={str(done).lower()} error=null")
            if done:
                break
        success = total_score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} score={total_score:.2f} rewards={rewards_str}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
        raise
    return total_score


if __name__ == "__main__":
    print("=" * 50)
    print("Email Triage - Inference Script")
    print("=" * 50)
    scores = []
    for task in TASKS:
        score = run_task(task)
        scores.append(score)
        print()
    avg = sum(scores) / len(scores)
    print(f"Average score: {avg:.2f}")
    sys.exit(0)
