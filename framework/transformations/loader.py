from typing import List

from dataloaders.data import Data
from transformations.transformations import Transform
from transformations.vanilla import Vanilla
from transformations.timestamp import Timestamp
from transformations.split import Split
from transformations.merge import Merge
from transformations.setproximity import SetProximity
from transformations.baseline import Baseline


class TransformLoader:
    def __init__(self):
        self.t_types = {
            "vanilla": Vanilla,
            "timestamp": Timestamp,
            "split": Split,
            "merge": Merge,
            "prox": SetProximity,
            "baseline": Baseline,
        }

    def get_t_types(self) -> List:
        return list(self.t_types.keys())

    def load(self, t_type: str, t_target: str, dataset: Data) -> Transform:
        if t_type in self.t_types.keys():
            target = self.t_types[t_type]
            return target.load(target, t_target, dataset)
        raise ValueError("Transformation type did not match any available options.")
