from typing import Dict, Tuple
from transformations.transformations import TransformTarget, Transform
from dataloaders.data import Data
from utils import Utils
import random as random


class Baseline(TransformTarget):
    def __init__(self):
        super().__init__()

    def get_entity_transform(self, data: Data) -> Transform:
        raise NotImplementedError

    def get_relation_transform(self, data: Data) -> Transform:
        return Relations(data)


class Relations(Transform):
    def __init__(self, data: Data):
        super().__init__(data)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)

        self.name_dict = {id: name for id, name in self.data.relations.items()}
        self.reverse_relation: Dict[int, int] = {id: id for id in self.data.relations.keys()}

        self.scoped_name_dict = {
            id: Utils.scope_name(self.name_dict[id], val["time_begin"], val["time_end"])
            for id, val in Utils.get_relation_timerange_all(self.data.triples).items()
        }

        self.init_parameters()
        random.seed(self.RANDOM_SEED)

    def get_name(self) -> str:
        return "baseline_" + self.VERSION + "_" + str(self.RANDOM_SEED)

    def init_parameters(self) -> None:
        self.VERSION = Utils.parse_helper(
            "Version", "uniform", lambda x: x if x in ["uniform", "weighted"] else ValueError
        )
        self.TARGET_SIZE = Utils.parse_helper("Target # predicates", 2 * len(self.data.relations), int)
        # first run done with '2312384781'
        self.RANDOM_SEED = Utils.parse_helper("Random seed: ", "noseed", int)

    def get_split(self) -> Tuple[int, int]:
        """ Get the next (predicate, timestamp) combination to split on. """
        if self.VERSION == "uniform":
            preds = self.data.triples.relation_id.unique()
            pred = preds[random.randint(0, len(preds) - 1)]
        elif self.VERSION == "weighted":
            # Weighted chooses a predicate from the currently existing facts.
            # As such, preds that occur more will also be chosen more frequently.
            row = self.data.triples.sample()
            pred = row.relation_id.values[0]  # uuuh??
        else:
            raise ValueError

        # Always pick timestamp uniformly.
        first, last = Utils.get_relation_timerange(self.data.triples, pred)
        timestamp = random.randint(first, last)
        return (pred, timestamp)

    # TODO: cleanup, this was copied from split.py
    def split_relation(self, relation_id: int, relation_group, split_time: float) -> Tuple[int, int]:
        """Split the relation in two:
         - Creates & returns IDs of the new relationship
         - Creates entries in the scoped name dictionary for the new relationships 
         """
        # Lookup the original relation for this relation
        original_id = self.reverse_relation[relation_id]

        # Name for the scoped relations
        relation_name = self.name_dict[original_id]
        time_data = relation_group.agg({"time_begin": "min", "time_end": "max"})

        # Get new ID's
        low_id = self.get_next_relation()
        high_id = self.get_next_relation()

        # Store reference to original relation
        self.reverse_relation[low_id] = original_id
        self.reverse_relation[high_id] = original_id

        # Assign scoped names to these ID's.
        self.scoped_name_dict[low_id] = Utils.scope_name(relation_name, time_data.time_begin, split_time)
        self.scoped_name_dict[high_id] = Utils.scope_name(relation_name, split_time, time_data.time_end)

        return low_id, high_id

    def transform(self) -> None:

        interval = int((self.TARGET_SIZE - len(self.data.relations)) / 10)
        while self.data.triples.relation_id.nunique() < self.TARGET_SIZE:
            pred, timestamp = self.get_split()

            # Get all facts for this predicate.
            groups = self.data.triples.groupby("relation_id")
            group = groups.get_group(pred)

            low_id, high_id = self.split_relation(pred, group, timestamp)
            change_data = {"split": timestamp, "low": low_id, "high": high_id}
            self.update_group(change_data, group, "relation")

            # Check at the end to prevents modulo by zero operation.
            current = self.data.triples.relation_id.nunique()
            if current % interval == 0:
                print(f"Currently at {current} out of {self.TARGET_SIZE} predicates.")

        self.data.relations = self.scoped_name_dict

