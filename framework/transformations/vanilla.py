import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from utils import Utils
from transformations.transformations import TransformTarget, Transform
from dataloaders.data import Data


class Vanilla(TransformTarget):
    def __init__(self):
        super().__init__()

    def get_entity_transform(self, data: Data) -> Transform:
        return Both(data)

    def get_relation_transform(self, data: Data) -> Transform:
        return Both(data)


class Both(Transform):
    """ Baseline dataset, just removes the temporal scope from all facts."""

    def __init__(self, data):
        super().__init__(data)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)
   
    def get_name(self) -> str:
        return "vanilla"

    def transform(self) -> None:
        pass

    def init_parameters(self) -> None:
        pass
