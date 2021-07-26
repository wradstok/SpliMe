import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from typing import Optional, List, Tuple, Dict, Union

from utils import Utils
from transformations.transformations import TransformTarget, Transform
from dataloaders.data import Data


class Timestamp(TransformTarget):
    def __init__(self):
        super().__init__()

    def get_entity_transform(self, data: Data) -> Transform:
        return Entities(data)

    def get_relation_transform(self, data: Data) -> Transform:
        return Relations(data)


class Entities(Transform):
    """Add temporal scope by adding a timestamp to each entity."""

    def __init__(self, data: Data):
        super().__init__(data)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)

        self.entity_timerange_dict = Utils.get_entity_timerange(self.data.triples)

        # Calculate new index for all entities
        self.entity_indices: IdMap = {x: {} for x in self.data.entities.keys()}
        self.reverse_entity_index = {}
        current_index = 0
        for entity_id in self.data.entities.keys():
            timerange = self.entity_timerange_dict[entity_id]
            for year in range(timerange["time_begin"], timerange["time_end"] + 1):
                self.entity_indices[entity_id][year] = current_index
                self.reverse_entity_index[current_index] = (year, entity_id)
                current_index += 1

    def get_name(self) -> str:
        return "entity_timestamped"

    def init_parameters(self) -> None:
        # No parameters required for this model.
        pass

    def transform(self) -> None:
        """Transform the dataset."""
        converted_triples = []
        entity_dict = {}

        #     0            1            2          3           4       5
        # [subject_id, relation_id, object_id, time_begin, time_end, source ]
        for row in self.data.triples.values:
            subject_id, object_id = row[0], row[2]

            subject_name = self.data.entities[subject_id]
            object_name = self.data.entities[object_id]

            # We create a new fact with timestamped entities for each year.
            for year in range(row[3], row[4] + 1):
                new_subject_id = self.entity_indices[subject_id][year]
                new_object_id = self.entity_indices[object_id][year]

                entity_dict[new_subject_id] = subject_name + "[" + str(year) + "]"
                entity_dict[new_object_id] = object_name + "[" + str(year) + "]"

                result = [new_subject_id, row[1], new_object_id, year, year, row[5]]
                converted_triples.append(result)

        self.data.triples = pd.DataFrame(
            converted_triples, columns=["subject_id", "relation_id", "object_id", "time_begin", "time_end", "source"],
        )
        self.data.entities = entity_dict


class Relations(Transform):
    """ Add temporal scope by applying a timestamp to each relation. """

    def __init__(self, data: Data):
        super().__init__(data)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)

        self.relations_timerange = Utils.get_relation_timerange_all(self.data.triples)

        # Calculate new index for all relations
        self.relation_indices: IdMap = {x: {} for x in self.data.relations.keys()}

        # Stores timestamped names for all relations.
        self.relation_dict = {}

        # Reverse relation index not required for here, but stored for merging approaches that
        # use timestamping as a pre-processing step.
        self.reverse_relation_index: Dict[int, Tuple[int, int]] = {}

        current_index = 0
        for source_rela_id in self.data.relations.keys():
            timerange = self.relations_timerange[source_rela_id]
            for year in range(timerange["time_begin"], timerange["time_end"] + 1):
                self.relation_indices[source_rela_id][year] = current_index

                self.relation_dict[current_index] = self.data.relations[source_rela_id] + "[" + str(year) + "]"
                self.reverse_relation_index[current_index] = (year, source_rela_id)
                current_index += 1

    def get_name(self) -> str:
        return "relation_timestamped"

    def init_parameters(self) -> None:
        # No parameters required for this model.
        pass

    def transform(self) -> None:
        """Transform the dataset."""
        converted_triples = []

        #     0            1            2          3           4       5
        # [subject_id, relation_id, object_id, time_begin, time_end, source ]
        for row in self.data.triples.itertuples():
            relation_id, source = row.relation_id, row.source

            for year in range(row.time_begin, row.time_end + 1):
                converted_triples.append(
                    [row.subject_id, self.relation_indices[relation_id][year], row.object_id, year, year, source,]
                )

        self.data.triples = pd.DataFrame(
            converted_triples, columns=["subject_id", "relation_id", "object_id", "time_begin", "time_end", "source",],
        )

        # Filter out anything that doesn't exist.
        existing_relation_ids = set(self.data.triples.relation_id.unique())
        self.reverse_relation_index = Utils.filter_dict_from_set(self.reverse_relation_index, existing_relation_ids)
        self.data.relations = Utils.filter_dict_from_set(self.relation_dict, existing_relation_ids)
