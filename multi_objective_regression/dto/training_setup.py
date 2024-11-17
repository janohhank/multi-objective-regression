from dataclasses import dataclass


@dataclass
class TrainingSetup:
    index: int
    features: list[str]
