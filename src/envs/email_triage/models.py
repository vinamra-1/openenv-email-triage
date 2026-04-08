from typing import Optional
from pydantic import BaseModel


class EmailAction(BaseModel):
    category: str


class EmailObservation(BaseModel):
    email_text: str = ""
    done: bool = False
    reward: Optional[float] = 0.0


class EmailState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
