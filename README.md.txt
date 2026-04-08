# 📧 Email Triage Environment

An RL environment where an AI agent learns to classify emails into SPAM, WORK, or PERSONAL categories.

## Real-World Task
Email overload is a genuine productivity problem. This environment trains agents to automatically triage emails, a task every knowledge worker faces daily.

## Tasks
| Task | Difficulty | Description |
|------|-----------|-------------|
| easy_triage | Easy | Obvious spam vs clear work/personal emails |
| medium_triage | Medium | Real-world emails requiring context understanding |
| hard_triage | Hard | Ambiguous emails that could fit multiple categories |

## Action Space
```json
{"category": "SPAM" | "WORK" | "PERSONAL"}
```

## Observation Space
```json
{"email_text": "string", "done": false, "reward": 0.0}
```

## Reward Function
- Correct classification: **1.0**
- Correct spam/non-spam but wrong subcategory: **0.2–0.5** (partial credit)
- Wrong: **0.0–0.1**

## Setup

### Docker
```bash
docker build -t email-triage .
docker run -p 8000:8000 email-triage
```

### Manual
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m uvicorn src.envs.email_triage.server.app:app --port 8000
```

## Inference
```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores
| Task | Score |
|------|-------|
| easy_triage | 0.85 |
| medium_triage | 0.72 |
| hard_triage | 0.55 |

## Author
Vinamra1