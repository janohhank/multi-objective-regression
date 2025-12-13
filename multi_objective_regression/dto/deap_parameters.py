import typing
from dataclasses import dataclass


@dataclass
class DeapParameters:
    iteration: typing.Optional[int]

    @staticmethod
    def from_dict(**data: dict) -> "DeapParameters":
        return DeapParameters(
            iteration=(data["iteration"] if "iteration" in data else None),
        )
