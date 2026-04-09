import os
from openai import OpenAI

# 1. Fetch variables
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "easy_triage")

# 2. Fix missing schemes (Critical for PyTorch Hackathon Validator)
if not API_BASE_URL.startswith(("http://", "https://")):
    API_BASE_URL = f"http://{API_BASE_URL}"

# 3. Debug Prints
print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[DEBUG] API_KEY={'set' if API_KEY != 'dummy-key' else 'dummy-key'}", flush=True)

# 4. Safe Initialization
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print("[INFO] OpenAI Client initialized successfully.", flush=True)
except Exception as e:
    print(f"[FATAL] OpenAI init failed! URL: {API_BASE_URL} | Error: {repr(e)}", flush=True)
    raise

# 5. OpenEnv required formatting
print(f"[START] task={TASK_NAME} env=email-triage model={MODEL_NAME}", flush=True)

# --- Add your specific LLM inference loop and logic below ---
# response = client.chat.completions.create(...)

