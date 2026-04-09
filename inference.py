import os
import sys
import re
import requests
from openai import OpenAI

SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS = 5

# Initialize client EXACTLY as the Scaler grader requested
try:
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
except KeyError:
    # Fallback ONLY for local testing if variables aren't set
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key="dummy_local_key"
    )

SYSTEM_PROMPT = "You are an Email Triage Assistant. Classify the email into exactly one of: [SPAM] [WORK] [PERSONAL]. Respond ONLY with the category in brackets."
TASKS = ["easy_triage", "medium_triage", "hard_triage"]

def reset_env(task):
    try:
        r = requests.post(f"{SERVER_URL}/reset", json={"task": task}, timeout=10)
        return r.json()
    except Exception as e:
        print(f"Env reset error: {e}")
        return {"observation": {"email_text": "Test email"}}

def step_env(category):
    try:
        r = requests.post(f"{SERVER_URL}/step", json={"category": category}, timeout=10)
        return r.json()
    except Exception as e:
        print(f"Env step error: {e}")
        return {"reward": 0.0, "done": True}

def get_llm_action(email_text):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Email: {email_text}\n\nReply with category in brackets:"},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\[(.*?)\]', text)
        return match.group(1).upper() if match else "SPAM"
    except Exception as e:
        # CRITICAL: Print the error so it shows up in the Scaler logs!
        print(f"API Error: {e}", file=sys.stderr)
        return "SPAM"

def run_task(task_name):
    all_rewards = []
    total_score = 0.0
    step_count = 0

    try:
        obs_data = reset_env(task_name)
        email_text = obs_data.get("observation", {}).get("email_text", "Test email")

        print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}")

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
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
        print(f"Error: {e}", file=sys.stderr)

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