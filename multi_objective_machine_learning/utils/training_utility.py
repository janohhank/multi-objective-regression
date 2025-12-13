import os
import typing
from abc import ABC

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


class TrainingUtility(ABC):

    @staticmethod
    def fit_standard_scaler(dataset: DataFrame) -> typing.Any:
        standard_scaler = StandardScaler()
        return standard_scaler.fit(dataset)

    @staticmethod
    def save_dataset(
        folder: str, name: str, dataset: tuple[DataFrame, DataFrame]
    ) -> typing.Any:
        dataset[1].to_frame().merge(
            dataset[0], left_index=True, right_index=True
        ).to_csv(os.path.join(folder, name) + ".csv", index=False, float_format="%.6f")
