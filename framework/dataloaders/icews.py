import numpy as np
import pandas as pd

from dataloaders.data import Data


class LoadICEWSData(Data):
    """ Load ICEWS datasets."""

    def __init__(self, options: dict):
        super().__init__(options)
        path = self.get_source_path()

        # Get data
        triples = self.get_data_with_source(path, ["subject_id", "relation_id", "object_id", "time_begin", "time_end"])

        # Find all unique year-month combinations, and assign an ordered id to each.
        # I.e. for n unique timestamps, the first timestamp get 0, the last gets n.
        self.unique_timestamp_set = {
            time: i for (i, time) in enumerate(sorted(set(map(self.parse_time, triples["time_begin"].unique()))))
        }

        # Create entity and relation dictionaries.
        entities = triples["subject_id"].append(triples["object_id"]).unique()
        reverse_entity_dict: Name2Id = {str(entities[x]): x for x in range(0, len(entities))}
        self.entities = {idx: name for name, idx in reverse_entity_dict.items()}

        relations = triples["relation_id"].unique()
        reverse_rela_dict: Name2Id = {relations[x]: x for x in range(0, len(relations))}
        self.relations = {idx: name for name, idx in reverse_rela_dict.items()}

        # Map the full names to ids.
        triples["subject_id"] = triples["subject_id"].map(reverse_entity_dict)
        triples["object_id"] = triples["object_id"].map(reverse_entity_dict)
        triples["relation_id"] = triples["relation_id"].map(reverse_rela_dict)

        self.triples = pd.DataFrame(
            data=np.apply_along_axis(self.fix_years, 1, triples.values), columns=triples.columns,
        )

        # Data is actually timestamped, so we pretend its an interval with minimal length.
        self.triples["time_end"] = self.triples["time_begin"]

        self.check_self()

        self.prepare()

    def from_paper(self) -> str:
        return "de-simple"

    def get_name(self) -> str:
        return "icews14"

    def parse_time(self, time: str) -> str:
        """ Converts year-month-day string to month-day string """ 
        return ("-").join(time.split("-")[1:3])

    def fix_years(self, numpy_row) -> np.ndarray:
        # Needs to become an integer
        index = self.parse_time(numpy_row[3])
        numpy_row[3] = self.unique_timestamp_set[index]
        return numpy_row
