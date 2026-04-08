import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import re
import requests
from transformers import AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

print("Starting Email Triage Training Script...")

SERVER_URL = "http://127.0.0.1:8000"
model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

system_prompt = "You are an expert Email Triage Assistant. Classify the email into exactly one of: [SPAM] [WORK] [PERSONAL]. Respond ONLY with the category in brackets."

def extract_guess(text):
    match = re.search(r'\[(.*?)\]', text)
    return match.group(1).upper() if match else "UNKNOWN"

def reward_func(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else ""
        guess = extract_guess(text)
        try:
            requests.post(f"{SERVER_URL}/reset", timeout=5)
            r = requests.post(f"{SERVER_URL}/step", json={"category": guess}, timeout=5)
            reward = float(r.json().get("reward", 0.0))
            print(f"  Guess: [{guess}] | Reward: {reward}")
        except Exception as e:
            print(f"  Env error: {e}")
            reward = 0.0
        rewards.append(reward)
    return rewards

emails = [
    "Congratulations! You won a $1000 Walmart gift card. Click here now.",
    "Hi team, please find attached the Q3 financial report for review.",
    "Hey honey, can you pick up some milk on your way home?",
    "URGENT: Your bank account has been compromised. Log in immediately.",
    "Are we still meeting for lunch at the cafe tomorrow?",
    "The server deployment is scheduled for tonight at 2 AM EST.",
    "You have been pre-selected for a FREE cruise. Limited slots!",
    "Can you review my PR before EOD? Blocked on your approval.",
    "Moms birthday is next week, dont forget to call her!",
] * 6

def make_prompt(email):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Email: {email}\n\nReply with category in brackets:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

dataset = Dataset.from_dict({"prompt": [make_prompt(e) for e in emails]})

grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    max_completion_length=15,
    num_generations=2,
    use_vllm=False,
    output_dir="email-triage-model",
    logging_steps=1,
    bf16=False,
    fp16=False,
    use_cpu=True,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model_name,
    processing_class=tokenizer,
    reward_funcs=[reward_func],
    train_dataset=dataset,
    args=grpo_config,
)

print("Starting RL Training Loop...")
trainer.train()
trainer.save_model("email-triage-model/final")
print("Training complete! Model saved.")
