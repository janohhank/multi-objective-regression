import typing
from abc import ABC

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


class TrainingUtility(ABC):
    @staticmethod
    def fit_standard_scaler(dataset: DataFrame) -> typing.Any:
        standard_scaler = StandardScaler()
        return standard_scaler.fit(dataset)
