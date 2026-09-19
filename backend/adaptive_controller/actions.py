"""Actions the AdaptiveController can choose between."""
from enum import Enum


class Action(str, Enum):
    RETRIEVE = "retrieve"
    STOP = "stop"