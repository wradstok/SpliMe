import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import math
import sys
import os
from multiprocessing import Pool

from my_types import Id, IdMap
from typing import Optional, List, Tuple, Dict, Union, Set

from utils import Utils
from transformations.transformations import TransformTarget, Transform
from transformations.timestamp import Timestamp
from dataloaders.data import Data


class Merge(TransformTarget):
    def __init__(self):
        super().__init__()

    def get_entity_transform(self, data: Data) -> Transform:
        return Entities(data)

    def get_relation_transform(self, data: Data) -> Transform:
        return Relations(data)


class MergeHelper:
    @staticmethod
    def transform(
        source_elements: dict,
        element_map: dict,
        elements: list,
        next_element_id: callable,
        triples: pd.DataFrame,
        reverse_element_index: dict,
        TARGET_SIZE: int,
    ):
        """ source_elements: Dictionary 
            element_map: either entity dictionary or relation dictionary
            elements: Either ['subject_id', 'object_id'] or ['relation_id']
            next_element_id: Function to get a fresh, new id for the element we are operating on.
            triples: Timestamped DataFrame to operate on.
            reverse_element_index: mapping from timestamped id to (timestamp, original_id)
        """
        # We use multiprocess to share the rolling sum calculations later.
        # We do this because this calculation is embarrassingly parallel, and we we need to perform it for
        # (possibly) thousands of columns. For smaller datasets though, it might not be worth the overhead.
        pool = Pool(os.cpu_count())

        # Convention:
        # 1) `source` refers to the original entity id's, from BEFORE they were timestamped.
        # 2) `target` refers to the timestamped entity id's.
        # 3) `updated` refers to the new entity id's which will replace the `target` id's.

        # Count # occurrences for each original element.
        counts: Dict[Id, dict] = {x: {} for x in source_elements.keys()}

        # Map each element->year combination to the timestamped version.
        src_to_target: IdMap = {x: {} for x in source_elements.keys()}

        # Contains for each updated id, a set of all targets.
        updated: Dict[Id, Set] = {x: {x} for x in element_map.keys()}

        # Count the #occurrences of each
        for element in elements:
            for target, count in triples.groupby(element).size().items():
                timestamp, source = reverse_element_index[target]
                counts[source][timestamp] = counts[source].get(timestamp, 0) + count
                src_to_target[source][timestamp] = target

        # Create dict where each entry contains a series whever every row is a timestamp.
        count_dfs = pd.DataFrame.from_dict(counts, orient="columns")
        count_dfs: Dict[int, pd.Series] = {
            idx: count_dfs[idx][count_dfs[idx].notna()].sort_index() for idx in count_dfs.columns
        }

        # Anything with less than 2 values cannot be merged, remove them.
        count_dfs: Dict[int, pd.Series] = {idx: values for idx, values in count_dfs.items() if len(values) >= 2}

        print("Starting merging")
        merged_count: int = 0
        while len(element_map) - merged_count > TARGET_SIZE:
            # Do rolling sum calculations independently for each entity.
            results = pool.starmap(Utils.roll_sum, count_dfs.items())

            # Find all items with the smallest # occurrences.
            # Note that because these items will always be the next to be merged (because this one just got larger!)
            # We do not need to perform our rolling sum operation again.
            _, min_val, _, _ = min(results, key=lambda x: x[1])
            results = [(item_id, start, end) for item_id, val, start, end in results if val == min_val]

            for (item_id, start_timestamp, end_timestamp) in results:
                # Check again to prevent overshooting our target
                if len(element_map) - merged_count < TARGET_SIZE:
                    break

                # It does not matter whether we update the first timestamp and remove the latter or vice-versa,
                # as long as we are consistent about it. Here, we set the first timestamp to be sum of both and
                # drop the latter. To slightly speed things up, we remove the entire entry from consideration
                # if there aren't enough values left.
                if len(count_dfs[item_id]) == 2:
                    del count_dfs[item_id]
                else:
                    count_dfs[item_id].at[start_timestamp] += count_dfs[item_id].at[end_timestamp]
                    count_dfs[item_id].drop(end_timestamp, inplace=True)

                # Set the new ID to all targets of both previous ID's. Delete entries for the previous ID's.
                new_id = next_element_id()

                start_target = src_to_target[item_id][start_timestamp]
                end_target = src_to_target[item_id][end_timestamp]
                updated[new_id] = updated[start_target].union(updated[end_target])

                del updated[start_target], updated[end_target]

                # Store targets for the new id, and delete the old targets (these ID's no longer exist).
                src_to_target[item_id][start_timestamp] = new_id
                del src_to_target[item_id][end_timestamp]

                # We have now reduced two relationships to one, that was easy :)
                merged_count += 1
                if merged_count % (int(len(element_map) / 50)) == 0:
                    print(f"Merged: {merged_count}, left: {len(element_map) - TARGET_SIZE - merged_count}")

        # Now we need to actually update the triples. To this this, we need to map each timestamped
        # triple to its target triple. We can get this map by reversing src_to_target,
        # mapping each element in the set to its dictionary key.
        update_map = {}
        item_names = {}
        for update_id, target_set in updated.items():
            lowest, highest = 99999, 0
            for target in target_set:
                update_map[target] = update_id

                # Figure out the first and last timestamp associated with this item (for naming purposes).
                timestamp, src_id = reverse_element_index[target]
                lowest = timestamp if timestamp < lowest else lowest
                highest = timestamp if timestamp > highest else highest

            item_names[update_id] = Utils.scope_name(source_elements[src_id], lowest, highest)

        return update_map, item_names


class Entities(Transform):
    def __init__(self, data):
        super().__init__(data)

        self.original_size = len(data.entities)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)

        self.source_entities = {idx: name for idx, name in self.data.entities.items()}

        # Timestamp the dataset.
        timestamp = Timestamp()
        timestamp = timestamp.get_entity_transform(data)
        timestamp.transform()
        self.reverse_entity_index = timestamp.reverse_entity_index

        # Re-initialize the max ids, because they have become incorrect due to the transformation.
        self.max_entity_id = max(self.data.entities.keys())
        self.max_relation_id = max(self.data.relations.keys())

        self.init_parameters()

    def get_name(self) -> str:
        return "merge_ent"

    def init_parameters(self) -> None:
        self.SHRINK_FACTOR = Utils.parse_helper("Shrink factor", 2, float)

        extra_size = len(self.data.entities) - self.original_size
        to_remove = int(extra_size / self.SHRINK_FACTOR)

        self.TARGET_SIZE = self.original_size + extra_size - to_remove
        print(f"Base size: {self.original_size}")
        print(f"Removing {to_remove} entities from excess {extra_size} target: {self.TARGET_SIZE}")

    def transform(self) -> None:
        print("Starting merge preparations")

        update_map, ent_names = MergeHelper.transform(
            self.source_entities,
            self.data.entities,
            ["subject_id", "object_id"],
            self.get_next_entity,
            self.data.triples,
            self.reverse_entity_index,
            self.TARGET_SIZE,
        )

        self.data.triples["subject_id"] = self.data.triples["subject_id"].map(update_map)
        self.data.triples["object_id"] = self.data.triples["object_id"].map(update_map)
        self.data.entities = ent_names


class Relations(Transform):
    def __init__(self, data: Data):
        super().__init__(data)

        self.original_size = len(data.relations)
        self.source_relations = {idx: name for idx, name in self.data.relations.items()}

        # First convert to timestamped dataset and get all reverse relations.
        timestamp = Timestamp()
        timestamp = timestamp.get_relation_transform(data)
        timestamp.transform()
        self.reverse_relation_index = timestamp.reverse_relation_index

        # Re-initialize the max ids, because they have become incorrect due to the transformation.
        self.max_entity_id = max(self.data.entities.keys())
        self.max_relation_id = max(self.data.relations.keys())

        self.init_parameters()

    def get_name(self) -> str:
        return "merge_rel"

    def init_parameters(self) -> None:
        self.SHRINK_FACTOR = Utils.parse_helper("Shrink factor", 2.0, float)

        extra_size = len(self.data.relations) - self.original_size
        to_remove = extra_size - int(extra_size / self.SHRINK_FACTOR)

        self.TARGET_SIZE = len(self.data.relations) - to_remove
        print(f"Base size: {self.original_size}")
        print(f"Removing {to_remove} relations from excess {extra_size} target: {self.TARGET_SIZE}")

    def transform(self) -> None:
        update_map, rela_names = MergeHelper.transform(
            self.source_relations,
            self.data.relations,
            ["relation_id"],
            self.get_next_relation,
            self.data.triples,
            self.reverse_relation_index,
            self.TARGET_SIZE,
        )

        self.data.triples["relation_id"] = self.data.triples["relation_id"].map(update_map)
        self.data.relations = rela_names
