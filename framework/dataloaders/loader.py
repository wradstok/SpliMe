from dataloaders.data import Data
from dataloaders.yago import LoadYago11kData
from dataloaders.wikidata import LoadWikiData12kData
from dataloaders.icews import LoadICEWSData

from typing import List

from functools import partial


class Dataloader:
    def __init__(self):
        self.datasets = {
            "yago11k": partial(LoadYago11kData),
            "icews14": partial(LoadICEWSData),
            "wikidata12k": partial(LoadWikiData12kData),
        }

    def get_datasets(self) -> List:
        return list(self.datasets.keys())

    def load(self, dataset: str, options: dict) -> Data:
        if dataset in self.datasets.keys():
            return self.datasets[dataset](options)
        raise ValueError("Dataset did not match any available options")
