import numpy as np
import pandas as pd

from utils import Utils
from my_types import Id, Name, Id2Name, Name2Id
from typing import Dict, Union, List, Any
from abc import ABC, abstractmethod
from pathlib import Path


class Data(ABC):
    """Abstract base class to load & parse a dataset"""

    def __init__(self, options: dict):
        self.options = options

        self.entities: Id2Name = {}
        self.relations: Id2Name = dict()
        self.triples = pd.DataFrame()
        self.buckets = {}

    def check_self(self) -> None:
        Utils.print_data_properties(self.triples)
        Utils.sanity_check(self.triples, self.entities, self.relations)
        Utils.print_time_info(self.triples)

    @abstractmethod
    def fix_years(self, row):
        """ Given a row, converts the time_begin and time_end in that row to integers and return the row."""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """ Name of data set"""
        raise NotImplementedError

    @abstractmethod
    def from_paper(self) -> str:
        """ Get the paper that created this data set."""
        raise NotImplementedError

    def get_data_with_source(self, path: str, columns: list):
        """Load the dataset with an extra column denoting whether the fact came from the train/test/validation set."""
        columns = columns + ["source"]
        train = pd.read_csv(path + "/train.txt", sep="\t", header=None, names=columns)
        test = pd.read_csv(path + "/test.txt", sep="\t", header=None, names=columns)
        valid = pd.read_csv(path + "/valid.txt", sep="\t", header=None, names=columns)

        # Set sources
        valid["source"] = valid["source"].apply(lambda x: "valid")
        test["source"] = test["source"].apply(lambda x: "test")
        train["source"] = train["source"].apply(lambda x: "train")

        complete = pd.concat([valid, test, train], axis="index", ignore_index=True)
        return complete

    def prepare(self):
        """ Cleanup the triples df, entity_dict and relation_dict so that they can be used for transformations."""
        # Remove rows where the temporal scope could not be parsed correctly.
        self.triples = self.triples[(self.triples.time_begin != -99999) & (self.triples.time_end != -99999)]

        # Remove rows with start time after the end time.
        time_diffs = self.triples["time_end"] - self.triples["time_begin"]
        invalid_times = time_diffs[time_diffs < 0].index
        self.triples.drop([x for x in invalid_times], inplace=True, axis="index")

        # Remove all entities that no longer have any facts associated with them from the dictionary.
        entity_counts = Utils.get_entity_counts(self.triples, self.entities)
        for idx, count in entity_counts.items():
            if count == 0:
                del self.entities[idx]

        # Remove all relations that no longer have any facts associated with them from the dictionary.
        relation_counts = Utils.get_group_counts(self.triples.groupby("relation_id"), self.relations)
        for idx, count in relation_counts.items():
            if count == 0:
                del self.relations[idx]

    def get_naive_size(self) -> int:
        """Get the number of triples in the naively converted version of this dataset."""
        return (self.triples.time_end - self.triples.time_begin + 1).sum()

    def get_source_path(self) -> str:
        # Go up three levels because we need to escape the framework directory
        path = Path(__file__).parent.parent.parent.absolute()
        target = str(path.joinpath("source_data/" + self.from_paper() + "/" + self.get_name()))
        return target

    def try_parse_year(self, year) -> Union[str, int]:
        # So, YAGO is a bit weird, and contains year strings like '195#', and after looking up that fact
        # it should instead be 1975.. So for any dates that do not parse as expected, we return -1, meaning invalid.
        # This is in line with HyTe, as they seem to ignore any dates that contain '#' and are not 4 digits long.
        if type(year) == int:
            return year

        explode = year.split("-")

        # Deal with years BC.
        parsed = "-" + explode[1] if year.startswith("-") else explode[0]

        if parsed == "####":
            return "weird"
        if parsed.find("#") == -1:
            return int(parsed)
        return -99999

    # Output the entity, relation dictionaries and the set of triples.
    def output(self, path: str) -> None:
        # Create path if not yet exists.
        Path(path).mkdir(parents=True, exist_ok=True)

        # Find all entities/relations in the dictionary that no longer occur, and remove them.
        empty_entities = [x for x, y in (Utils.get_entity_counts(self.triples, self.entities)).items() if y == 0]
        empty_relations = [
            x
            for x, y in (Utils.get_group_counts(self.triples.groupby("relation_id"), self.relations)).items()
            if y == 0
        ]

        for x in empty_entities:
            del self.entities[x]
        for x in empty_relations:
            del self.relations[x]

        entity_df = Utils.df_from_name_dict(self.entities)
        rela_df = Utils.df_from_name_dict(self.relations)

        # Output a time map.
        if len(self.buckets) > 0:
            with open(path + "/time_map.dict", "w") as f:
                for key, bucket in self.buckets.items():
                    f.write(f"{key}\t{bucket['min']}\t{bucket['max']}\n")

        # In case we want to run our data set under HyTE code, we should keep the ID's rather than
        # convert to named entities. Since we have removed some entities, we need to re-index the dataset.
        if self.options["nonames"]:
            # Convert the dictionaries to gapless
            entity_dict = entity_df.to_dict(orient="series")["name"]
            rela_dict = rela_df.to_dict(orient="series")["name"]

            # Map old ID's to new ID's
            reverse_entity_dict = {name: idx for idx, name in entity_dict.items()}
            entity_map = {
                idx: reverse_entity_dict[self.entities[idx]]
                for idx in self.triples.subject_id.append(self.triples.object_id).unique()
            }

            reverse_rela_dict = {name: idx for idx, name in rela_dict.items()}
            rela_map = {idx: reverse_rela_dict[self.relations[idx]] for idx in self.triples.relation_id.unique()}

            # Apply
            self.triples.subject_id = self.triples.subject_id.map(entity_map)
            self.triples.relation_id = self.triples.relation_id.map(rela_map)
            self.triples.object_id = self.triples.object_id.map(entity_map)
        else:
            # Map to names
            self.triples["subject_id"] = self.triples["subject_id"].map(self.entities)
            self.triples["relation_id"] = self.triples["relation_id"].map(self.relations)
            self.triples["object_id"] = self.triples["object_id"].map(self.entities)

        # Recreate entity_df and rela_df, because the entity/relation sets may have changed during post-processing.
        entity_df = Utils.df_from_name_dict(self.entities)
        rela_df = Utils.df_from_name_dict(self.relations)

        valid = self.triples[self.triples["source"] == "valid"]
        test = self.triples[self.triples["source"] == "test"]
        train = self.triples[self.triples["source"] == "train"]

        cols = ["subject_id", "relation_id", "object_id"]

        # Remove any triples from valid/test that occur in the training set (these could appear due to re-occurring quintuples).
        if self.options["filter_dupes"] == "both" or self.options["filter_dupes"] == "intra":
            # Remove duplicates from the all data sets.
            valid = valid.drop_duplicates(cols, ignore_index=True)
            test = test.drop_duplicates(cols, ignore_index=True)
            train = train.drop_duplicates(cols, ignore_index=True)

        if self.options["filter_dupes"] == "both" or self.options["filter_dupes"] == "inter":
            # Remove triples from the valid/test set that are already in the training set.
            train_triples = {(s, p, o) for s, p, o in train[cols].values}
            valid = pd.DataFrame(
                [
                    row
                    for index, row in valid.iterrows()
                    if (row.subject_id, row.relation_id, row.object_id) not in train_triples
                ]
            )
            test = pd.DataFrame(
                [
                    row
                    for index, row in test.iterrows()
                    if (row.subject_id, row.relation_id, row.object_id) not in train_triples
                ]
            )

        # Update triples to match new circumstances, so that its data can be safely for the summary output.
        self.triples = pd.concat([valid, test, train], axis="index", ignore_index=True)

        # Output entity/relation dictionaries for
        entity_df.to_csv(path + "/entities.dict", sep="\t", header=False, columns=["name"])
        rela_df.to_csv(path + "/relations.dict", sep="\t", header=False, columns=["name"])

        # Recreate the train/test/valid split.
        valid.to_csv(path + "/valid.txt", sep="\t", columns=cols, header=False, index=False)
        test.to_csv(path + "/test.txt", sep="\t", columns=cols, header=False, index=False)
        train.to_csv(path + "/train.txt", sep="\t", columns=cols, header=False, index=False)
