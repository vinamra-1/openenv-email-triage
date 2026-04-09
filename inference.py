import os
import sys
import re
import requests
from openai import OpenAI

SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
MAX_STEPS = 5

SYSTEM_PROMPT = "You are an Email Triage Assistant. Classify the email into exactly one of: [SPAM] [WORK] [PERSONAL]. Respond ONLY with the category in brackets."

TASKS = ["easy_triage", "medium_triage", "hard_triage"]


def reset_env(task):
    try:
        r = requests.post(f"{SERVER_URL}/reset", json={"task": task}, timeout=20)
        return r.json()
    except Exception:
        return {"observation": {"email_text": "Test email"}}


def step_env(category):
    try:
        r = requests.post(f"{SERVER_URL}/step", json={"category": category}, timeout=20)
        return r.json()
    except Exception:
        return {"reward": 0.0, "done": True}


def get_llm_action(email_text):
    text = email_text.lower()

    # 🔥 STRONG RULE-BASED SYSTEM (covers most cases)

    spam_keywords = [
        "lottery", "prize", "winner", "free", "money", "urgent", "click",
        "offer", "buy now", "discount", "limited time", "cash", "reward",
        "claim", "bonus", "congratulations", "selected", "exclusive",
        "deal", "win", "gift", "voucher", "promo", "sale", "cheap",
        "earn", "income", "investment", "crypto", "bitcoin", "loan",
        "credit", "risk free", "guarantee"
    ]

    work_keywords = [
        "meeting", "project", "deadline", "client", "report", "schedule",
        "team", "manager", "update", "work", "invoice", "presentation",
        "review", "task", "assignment", "office", "company", "business",
        "plan", "proposal", "budget", "analysis", "training", "job",
        "interview", "hr", "policy", "documentation"
    ]

    personal_keywords = [
        "family", "friend", "party", "dinner", "home", "trip",
        "birthday", "hangout", "call me", "mom", "dad", "bro",
        "sister", "love", "wedding", "festival", "vacation",
        "weekend", "picnic", "celebration", "invite", "gathering",
        "hello", "hi", "how are you"
    ]

    # ✅ RULE MATCHING
    if any(word in text for word in spam_keywords):
        return "SPAM"

    if any(word in text for word in work_keywords):
        return "WORK"

    if any(word in text for word in personal_keywords):
        return "PERSONAL"

    # 🔁 FALLBACK TO LLM (only if needed)
    try:
        if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
            client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"]
            )
        else:
            client = OpenAI(
                base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
                api_key=os.environ.get("HF_TOKEN", "dummy")
            )

        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

        print("[DEBUG] Calling LLM API...", file=sys.stderr)

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": email_text}
            ],
            max_output_tokens=10,
            temperature=0.0,
        )

        output = response.output[0].content[0].text.strip().upper()

        # ✅ STRONG PARSING
        if "SPAM" in output:
            return "SPAM"
        elif "WORK" in output:
            return "WORK"
        elif "PERSONAL" in output:
            return "PERSONAL"

        match = re.search(r'\[(.*?)\]', output)
        return match.group(1).upper() if match else "SPAM"

    except Exception as e:
        print(f"[ERROR] API Error: {e}", file=sys.stderr)
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
        print(f"[END] success=false steps=0 score=0.00 rewards=0.00")
        print(f"[ERROR] {e}", file=sys.stderr)

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