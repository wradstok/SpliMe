import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import math
from collections import OrderedDict
from my_types import Id, Name, Id2Name, Name2Id
from typing import Optional, List, Set, Tuple, Dict, Union

from utils import Utils
from transformations.transformations import TransformTarget, Transform
from dataloaders.data import Data


class Split(TransformTarget):
    def __init__(self):
        super().__init__()

    def get_entity_transform(self, data: Data) -> Transform:
        return Entities(data)

    def get_relation_transform(self, data: Data) -> Transform:
        return Relations(data)


class Split_Helper:
    @staticmethod
    def split_time(selection: pd.DataFrame) -> int:
        """Get the year that splits the timerange in the selection in the middle."""
        timerange = selection.agg({"time_begin": "min", "time_end": "max"})
        return int((timerange["time_begin"] + timerange["time_end"]) / 2)

    @staticmethod
    def split_occurrences(selection: pd.DataFrame) -> int:
        """Split so that the # of triples on each side of the split is equal."""
        # For each year, get the # of facts that end before the year and the # of facts that begin after it.
        # We do not need to count the facts that are active during the split, because they count on both sides.
        # We want to find the most balanced year.
        time_dict = OrderedDict()
        for time in np.sort(selection["time_begin"].append(selection["time_end"]).unique(), axis=0):
            time_dict[time] = {"end_before": 0, "start_after": 0}

        # Count for every time how many facts have ended/begun at that point. Cumulative sum of sorts
        prev_edited_time = selection["time_end"].min()
        for idx, row in selection.sort_values(by="time_end", axis="index").iterrows():
            time_dict[row.time_end]["end_before"] = time_dict[prev_edited_time]["end_before"] + 1
            prev_edited_time = row.time_end

        prev_edited_time = selection["time_begin"].max()
        for idx, row in selection.sort_values(by="time_begin", axis="index", ascending=False).iterrows():
            time_dict[row.time_begin]["start_after"] = time_dict[prev_edited_time]["start_after"] + 1
            prev_edited_time = row.time_begin

        # Some times will only occur as a start or end, so these will have the other value set to 0.
        # For end_before, we can set it to the time before. This can be done in 1 pass because dicts maintain insertion order.
        options = ["end_before", "start_after"]
        previous_time = selection["time_end"].min()
        for time, val in time_dict.items():
            for option in options:
                if val[option] == 0:
                    time_dict[time][option] = time_dict[previous_time][option]
            previous_time = time

        scores = {key: abs(val["end_before"] - val["start_after"]) for key, val in time_dict.items()}

        # Remove first & last year from consideration as splitting on these makes no sense!
        # However, now it can happen that the scores are empty, this is the case when every fact starts or ends at the same
        # two points in the time. In this case, we split on/after the begin_time.
        del scores[selection["time_begin"].min()], scores[selection["time_end"].max()]
        if len(scores) == 0:
            print("Fell back to time split")
            return Split_Helper.split_time(selection)
        return min(scores, key=scores.get)


class Entities(Transform):
    """ Add temporal scope by splitting entities/ adding splits for entities."""

    def __init__(self, data: Data):
        super().__init__(data)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)

        self.init_parameters()

    def get_name(self) -> str:
        return "entity_split_" + self.SPLIT_METHOD

    def evaluate_stop_condition(self) -> bool:
        if self.MEASURE_METHOD == "triples":
            return len(self.data.triples) < self.TARGET_SIZE
        elif self.MEASURE_METHOD == "entities":
            return len(self.data.entities) < self.TARGET_SIZE
        else:
            raise NotImplementedError

    def init_parameters(self) -> None:
        self.MEASURE_METHOD = "entities"

        self.GROW_FACTOR = Utils.parse_helper("Grow factor", 2.0, float)
        self.TARGET_SIZE = math.ceil(len(self.data.entities) * self.GROW_FACTOR)
        print(f"Extending to {self.TARGET_SIZE} entities, from {len(self.data.entities)} entities")

        self.SPLIT_METHOD = Utils.parse_helper(
            "Split method", "time", lambda x: x if x == "time" or x == "count" else ValueError,
        )

    def transform(self) -> None:
        # Get the # occurrences of each entity (as either subject or object).
        print("Starting preparations")

        # Gather all facts for each entity.
        slices = {
            ent: self.data.triples[(self.data.triples.subject_id == ent) | (self.data.triples.object_id == ent)]
            for ent in self.data.entities.keys()
        }

        # Count the total # occurences and the total timerange for each entity.
        ent_counts = pd.Series({ent: len(ent_slice) for ent, ent_slice in slices.items()})
        ent_timerange = Utils.get_entity_timerange(self.data.triples)
        unique_timesteps = {
            ent: slices[ent].time_begin.append(slices[ent].time_end).nunique() for ent in self.data.entities.keys()
        }

        # Create a target map for each (entity,timestamp) combination. Initial value is to keep the id the same.
        targets = {
            ent: {time: ent for time in range(ent_timerange[ent]["time_begin"], ent_timerange[ent]["time_end"] + 1)}
            for ent in self.data.entities.keys()
        }

        split_times = {x: set() for x in self.data.entities.keys()}

        # Before starting, remove any items from consideration that do not have enough room to split.
        # This is the case if there are less than 3 unique timesteps. Because then all facts have the same times,
        # and thus it makes no sense to split.
        invalids: Set[int] = {ent for ent, time in ent_timerange.items() if unique_timesteps[ent] <= 2}
        ent_counts = ent_counts.drop(invalids).sort_values(ascending=False)

        # Also drop any entities which only occur once, these will not be interesting to split on.
        ent_counts = ent_counts[ent_counts > 1]

        # Store the original Id for each new ID.
        reverse_targets = {ent: ent for ent in self.data.entities.keys()}
        print("Starting splitting")
        while len(ent_timerange) < self.TARGET_SIZE:
            if len(ent_counts) == 0:
                print("Stopped splitting because there were no entities left to split on.")
                break

            if (self.TARGET_SIZE - len(ent_timerange)) % 50 == 0:
                print(f"Currently at {len(ent_timerange)} out of {self.TARGET_SIZE} entities")

            # Get entity to split and then remove it.
            ent_to_split = ent_counts.idxmax()
            original_ent = reverse_targets[ent_to_split]
            ent_slice = slices[original_ent]
            ent_counts.drop(ent_to_split, inplace=True)

            # Selection contains all the facts that will be updated to contain the fact we will be splitting.
            # It is created by selecting all facts from the dataframe containing the original fact, and then
            # slicing the appropiate time period.
            first_time, last_time = (
                ent_timerange[ent_to_split]["time_begin"],
                ent_timerange[ent_to_split]["time_end"],
            )
            selection = ent_slice[(ent_slice.time_begin >= first_time) & (ent_slice.time_end <= last_time)]
            if self.SPLIT_METHOD == "time":
                split_time = Split_Helper.split_time(selection)
            elif self.SPLIT_METHOD == "count":
                split_time = Split_Helper.split_occurrences(selection)
            else:
                raise ValueError

            split_times[original_ent].add(split_time)

            # Create the new entities.
            low_id, high_id = self.get_next_entity(), self.get_next_entity()
            split_times[low_id], split_times[high_id] = set(), set()
            reverse_targets[low_id], reverse_targets[high_id] = original_ent, original_ent

            # Update ent_timerange
            ent_timerange[low_id] = {"time_begin": first_time, "time_end": split_time}
            ent_timerange[high_id] = {"time_begin": split_time + 1, "time_end": last_time}

            # Update targets
            for time in range(first_time, split_time + 1):
                targets[original_ent][time] = low_id

            for time in range(split_time + 1, last_time + 1):
                targets[original_ent][time] = high_id

            del ent_timerange[ent_to_split]

            # Add the new IDs to the ent_counts if they meet the # timestamps and the # of facts criteria.
            ids = [low_id, high_id]
            for curr_id in ids:
                begin, end = ent_timerange[curr_id]["time_begin"], ent_timerange[curr_id]["time_end"]
                new_selection = selection[(selection.time_begin >= begin) & (selection.time_end <= end)]
                if len(new_selection) > 1 and new_selection.time_begin.append(new_selection.time_end).nunique() > 2:
                    ent_counts = ent_counts.append(pd.Series([len(new_selection)], index=[curr_id]))

        # Now that we are done, perform the actual update.
        print("Updating rows...")

        # Create new entity names.
        ent_names = {
            ent: Utils.scope_name(
                self.data.entities[reverse_targets[ent]], timerange["time_begin"], timerange["time_end"]
            )
            for ent, timerange in ent_timerange.items()
        }

        converted_triples = []
        for row in self.data.triples.itertuples():
            # Check whether the subject or object has one or more splits during this range
            all_splits = sorted(list(split_times[row.subject_id].union(split_times[row.object_id])))
            start = np.searchsorted(all_splits, row.time_begin)
            end = np.searchsorted(all_splits, row.time_end)

            all_splits = all_splits[start:end]

            # Create a fact from the previous split to the current one,
            # fetching the correct ID's in the process. We manually add row.time_end to the list of splits
            # to ensure the fact is complete.
            prev_split = row.time_begin
            if len(all_splits) == 0 or all_splits[-1] != row.time_end:
                all_splits.append(row.time_end)

            for split in all_splits:
                converted_triples.append(
                    [
                        targets[row.subject_id][split],
                        row.relation_id,
                        targets[row.object_id][split],
                        prev_split,
                        split,
                        row.source,
                    ]
                )

        self.data.triples = pd.DataFrame(data=converted_triples, columns=self.data.triples.columns)
        self.data.entities = ent_names


class Relations(Transform):
    """ Add temporal scope by adding splits to relations"""

    def __init__(self, data: Data):
        super().__init__(data)

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)
        
        # name_dict contains the basic (unscoped name) for all relations.
        self.name_dict: Id2Name = {idx: name for idx, name in self.data.relations.items()}

        # scoped_name_dict contains relation names scoped to their lifespan.
        self.scoped_name_dict: Id2Name = {
            idx: Utils.scope_name(self.name_dict[idx], val["time_begin"], val["time_end"])
            for idx, val in Utils.get_relation_timerange_all(self.data.triples).items()
        }

        # reverse_relation stores for each relation what its original source relation is.
        # starts of with just pointing to itself.
        self.reverse_relation: Dict[Id, Id] = {idx: idx for idx in self.data.relations.keys()}

        self.init_parameters()

    def get_name(self) -> str:
        return "relation_split_" + self.SPLIT_METHOD

    def init_parameters(self) -> None:
        self.MEASURE_METHOD = Utils.parse_helper(
            "Measure method", "triples", lambda x: x if x == "triples" or x == "relations" else ValueError,
        )
        if self.MEASURE_METHOD == "triples":
            self.SHRINK_FACTOR = Utils.parse_helper("Shrink factor", 0.01, float)
            self.TARGET_SIZE = len(self.data.triples) + math.ceil(
                self.SHRINK_FACTOR * (self.data.get_naive_size() - len(self.data.triples))
            )
            print(f"Shrinking to {self.TARGET_SIZE} triples, from {len(self.data.triples)} triples")
        else:
            self.GROW_FACTOR = Utils.parse_helper("Grow factor", 2.0, float)
            self.TARGET_SIZE = math.ceil(len(self.data.relations) * self.GROW_FACTOR)
            print(f"Extending to {self.TARGET_SIZE} relations, from {len(self.data.relations)} relations")

        self.SPLIT_METHOD = Utils.parse_helper(
            "Split method", "time", lambda x: x if x == "time" or x == "count" else ValueError,
        )

    def split_relation(self, relation_id: int, relation_group, split_time: float) -> Tuple[int, int]:
        """Split the relation in two:
         - Creates & returns IDs of the new relationship
         - Creates entries in the scoped name dictionary for the new relationships 
         """
        # Lookup the original relation for this relation
        original_id = self.reverse_relation[relation_id]

        # Name for the scoped relation
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

    def evaluate_stop_condition(self):
        if self.MEASURE_METHOD == "triples":
            return len(self.data.triples) < self.TARGET_SIZE
        elif self.MEASURE_METHOD == "relations":
            return self.data.triples["relation_id"].nunique() < self.TARGET_SIZE
        else:
            raise NotImplementedError

    def transform(self) -> None:
        while self.evaluate_stop_condition():
            curr_count = self.data.triples["relation_id"].nunique()
            if (self.TARGET_SIZE - curr_count) % 50 == 0:
                print("Currently at " + str(curr_count) + " out of " + str(self.TARGET_SIZE) + " relations")

            all_rela_groups = self.data.triples.groupby("relation_id")

            # Valid relationships must have
            # 1) A span of at least 3 years over all facts, to ensure that there is room to split.
            # 2) Not all facts in it should start or end at the same year (otherwise all facts will end up in the same positions)
            valid_relations = all_rela_groups.agg({"time_begin": "min", "time_end": "max"})
            valid_relations = {
                x for x in valid_relations[valid_relations.time_end - valid_relations.time_begin >= 3].index
            }
            valid_relations = valid_relations.intersection(
                {idx for idx, val in all_rela_groups["time_begin"].nunique().items() if val > 2}
            )
            valid_relations = valid_relations.intersection(
                {idx for idx, val in all_rela_groups["time_end"].nunique().items() if val > 2}
            )
            if len(valid_relations) == 0:
                print("Exiting transformation before target was reached, because there is nothing left to split.")
                break

            # From these, get the relation with the currently(!) highest # occurrences.
            rela_to_split_id, max_count = -1, -1
            for idx in valid_relations:
                count = len(all_rela_groups.get_group(idx))
                if count > max_count:
                    rela_to_split_id, max_count = idx, count

            rela_group = all_rela_groups.get_group(rela_to_split_id)

            # What time do we split the relation on?
            if self.SPLIT_METHOD == "count":
                split_year = Split_Helper.split_occurrences(rela_group)
            elif self.SPLIT_METHOD == "time":
                split_year = Split_Helper.split_time(rela_group)
            else:
                raise ValueError

            low_id, high_id = self.split_relation(rela_to_split_id, rela_group, split_year)

            change_data = {"split": split_year, "low": low_id, "high": high_id}

            self.update_group(change_data, rela_group, "relation")

        self.data.relations = self.scoped_name_dict
