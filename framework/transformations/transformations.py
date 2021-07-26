import numpy as np
import pandas as pd
import math

from typing import List, Union
from pathlib import Path
from abc import ABC, abstractmethod

from dataloaders.data import Data


class Transform(ABC):
    """Abstract base class to transform a dataset"""

    def __init__(self, data: Data):
        self.data = data
        self.max_entity_id: int = max(self.data.entities.keys())
        self.max_relation_id: int = max(self.data.relations.keys())

        self.MEASURE_METHOD: str = "N/A"
        self.SEARCH_METHOD: str = "N/A"
        self.GROW_FACTOR: float = 0
        self.SHRINK_FACTOR: float = 0
        self.TARGET_SIZE: int = 0
        self.EPSILON_FACTOR: int = 0
        self.DISTANCE_METRIC: str = "N/A"
        self.RANDOM_SEED: Union[int, str] = "N/A"
        self.VERSION: str = "N/A"

    @abstractmethod
    def transform(self) -> None:
        """Transform the dataset. """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self) -> None:
        raise NotImplementedError

    def get_next_entity(self) -> int:
        self.max_entity_id += 1
        return self.max_entity_id

    def get_next_relation(self) -> int:
        self.max_relation_id += 1
        return self.max_relation_id

    def get_dest_path(self) -> str:
        # Just like reading the data, we need to go up three levels to escape the framework directory.
        path = Path(__file__).parent.parent.parent.absolute()
        data_src = self.data.get_name()
        transform_method = self.get_name()

        target = str(path.joinpath("KnowledgeGraphEmbedding/transformed_data/" + data_src + "_" + transform_method))

        return target

    def create_rowpart(self, updated_id: int, row: np.ndarray, position: str) -> List[int]:
        """ Insert the updated_id in the correct spot, position can be 'subject', 'object' or 'relation'  """
        if position == "subject":
            return [updated_id, row.relation_id, row.object_id]
        elif position == "object":
            return [row.subject_id, row.relation_id, updated_id]
        elif position == "relation":
            return [row.subject_id, updated_id, row.object_id]
        else:
            raise ValueError

    def update_group(self, cda, group, position: str) -> None:
        """ Given split information, a group to split, and the element being changed,
             recreate all rows in the group to reflect the new information."""
        # Create the new rows.
        converted_triples = []

        for _, row in group.iterrows():
            low = self.create_rowpart(cda["low"], row, position)
            high = self.create_rowpart(cda["high"], row, position)

            # Is the fact active during the split?
            if row.time_begin < cda["split"] < row.time_end:
                converted_triples.append(low + [row.time_begin, math.floor(cda["split"]), row.source])
                converted_triples.append(high + [math.ceil(cda["split"]), row.time_end, row.source])
            # Fact ends before or on the split, assign low id.
            elif row.time_end <= cda["split"]:
                converted_triples.append(low + [row.time_begin, row.time_end, row.source])
            # Fact after the split? Assign high id.
            else:
                converted_triples.append(high + [row.time_begin, row.time_end, row.source])

        # Drop the previously existing rows.
        self.data.triples.drop([x for x in group.index], inplace=True, axis="index")

        # Add the new rows (this breaks the previous index).
        self.data.triples = self.data.triples.append(
            pd.DataFrame(data=converted_triples, columns=self.data.triples.columns), ignore_index=True,
        )


class TransformTarget(ABC):
    """ Class to target entity/relation"""

    def __init__(self):
        pass

    @abstractmethod
    def get_entity_transform(self, data: Data) -> Transform:
        raise NotImplementedError

    @abstractmethod
    def get_relation_transform(self, data: Data) -> Transform:
        raise NotImplementedError

    def load(self, t_target: str, data: Data) -> Transform:
        if t_target == "entities":
            return self.get_entity_transform(self, data)
        elif t_target == "relations":
            return self.get_relation_transform(self, data)
        raise ValueError("Transformation target did not match available options")
