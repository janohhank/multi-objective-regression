import typing
from dataclasses import dataclass


@dataclass
class DeapParameters:
    iteration: typing.Optional[int] = None

    @staticmethod
    def from_dict(**data: str) -> "DeapParameters":
        return DeapParameters(
            iteration=(data["iteration"] if "iteration" in data else None),
        )
