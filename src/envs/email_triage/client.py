from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import EmailAction, EmailObservation, EmailState


class EmailEnv(EnvClient[EmailAction, EmailObservation, EmailState]):

    def _parse_observation(self, payload: dict) -> EmailObservation:
        return EmailObservation(**payload)

    def _parse_state(self, payload: dict) -> EmailState:
        return EmailState(**payload)

    def _serialize_action(self, action: EmailAction) -> dict:
        return action.model_dump()