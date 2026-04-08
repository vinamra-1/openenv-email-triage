import random
import uuid
from typing import Optional
from openenv.core.env_server import Environment
from ..models import EmailAction, EmailObservation, EmailState

EMAILS = [
    {"text": "Congratulations! You won a $1000 Walmart gift card. Click here to claim now.", "label": "SPAM"},
    {"text": "Hi team, please find attached the Q3 financial report. We discuss at 3 PM.", "label": "WORK"},
    {"text": "Hey honey, can you pick up some milk and eggs on your way home?", "label": "PERSONAL"},
    {"text": "URGENT: Your bank account has been compromised. Log in immediately.", "label": "SPAM"},
    {"text": "Just checking in, are we still meeting for lunch at the cafe tomorrow?", "label": "PERSONAL"},
    {"text": "The server deployment is scheduled for tonight at 2 AM EST. Please merge all code.", "label": "WORK"},
    {"text": "You have been pre-selected for a FREE cruise. Limited slots available!", "label": "SPAM"},
    {"text": "Can you review my PR before EOD? Blocked on your approval.", "label": "WORK"},
    {"text": "Mom's birthday is next week, don't forget to call her!", "label": "PERSONAL"},
]

class EmailEnvironment(Environment[EmailAction, EmailObservation, EmailState]):

    def __init__(self):
        super().__init__()
        self._state = EmailState(episode_id=str(uuid.uuid4()), step_count=0)
        self.current_email = None

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> EmailObservation:
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self.current_email = random.choice(EMAILS)
        return EmailObservation(
            email_text=self.current_email["text"],
            done=False,
            reward=0.0,
        )

    async def reset_async(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> EmailObservation:
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(self, action: EmailAction, timeout_s: Optional[float] = None, **kwargs) -> EmailObservation:
        self._state.step_count += 1
        guess = action.category.upper().strip()
        correct_label = self.current_email["label"]
        reward = 1.0 if guess == correct_label else 0.0
        return EmailObservation(
            email_text=self.current_email["text"],
            done=True,
            reward=reward,
        )

    async def step_async(self, action: EmailAction, timeout_s: Optional[float] = None, **kwargs) -> EmailObservation:
        return self.step(action, timeout_s=timeout_s, **kwargs)

    def close(self) -> None:
        pass

    @property
    def state(self) -> EmailState:
        return self._state