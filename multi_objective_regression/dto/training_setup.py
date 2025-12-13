from dataclasses import dataclass


@dataclass
class TrainingSetup:
    index: int
    # Selected features' name
    features: list[str]
    # Target feature name
    target_feature: str
