from dataclasses import dataclass


@dataclass
class DeapParameters:
    initial_population_size: int
    iteration: int

    @staticmethod
    def from_dict(**data: dict) -> "DeapParameters":
        return DeapParameters(
            initial_population_size=(
                data["initial_population_size"]
                if "initial_population_size" in data
                else None
            ),
            iteration=(data["iteration"] if "iteration" in data else None),
        )
