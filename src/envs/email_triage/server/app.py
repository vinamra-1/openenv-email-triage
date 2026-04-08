import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from openenv.core.env_server import create_fastapi_app
from .environment import EmailEnvironment
from ..models import EmailAction, EmailObservation

# Pass the CLASS, not an instance
app = create_fastapi_app(EmailEnvironment, EmailAction, EmailObservation)