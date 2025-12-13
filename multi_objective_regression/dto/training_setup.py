from dataclasses import dataclass


@dataclass
class TrainingSetup:
    index: int
    # Selected features' name
    features: list[str]
    # Target feature name
    target_feature: str

    @staticmethod
    def from_dict(**data: dict) -> "TrainingSetup":
        return TrainingSetup(
            index=(data["index"] if "index" in data else None),
            features=(data["features"] if "features" in data else []),
            target_feature=(
                data["target_feature"] if "target_feature" in data else None
            ),
        )
