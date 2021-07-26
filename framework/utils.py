import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import math
import sys
from my_types import Id, Name, Time, Id2Name, Name2Id
from typing import Callable, Tuple, Any, Dict


class Utils:
    @staticmethod
    def aggregate_years(triples: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """ Aggregate fact timestamps into timespans s.t. that there is a minimum number of facts for each timespan.
            A fact is said to be inside a timespan if its valid-time (partially) overlaps the given timespan. 
            As such, a fact can be counted towards multiple buckets. """
        min_time, max_time = triples.time_begin.min(), triples.time_end.max()
        min_size = 300 # As in HyTE and other literature.

        buckets = {0: {"min": min_time, "max": min_time}}  # All buckets and their first and last year.

        while buckets[list(buckets.keys())[-1]]["max"] <= max_time:
            bucket = buckets[list(buckets.keys())[-1]]

            # Keep adding years as long as either
            # 1) We have less than min_facts in the current bucket slice.
            #    Note: the facts valid time is inside the range defined by the bucket.
            #    I.e. (min <= time_begin <= max) OR (min <= time_end <= max)
            # 2) Closing the bucket here would result in less than min_facts facts left to fill the next bucket with.
            #    These will be the facts that END after the current max: I.e. time_end >= max.
            while (
                len(
                    triples.loc[
                        ((triples.time_begin >= bucket["min"]) & (triples.time_begin <= bucket["max"]))
                        | ((triples.time_end >= bucket["min"]) & (triples.time_end <= bucket["max"]))
                    ]
                )
                < min_size
                or len(triples.loc[triples.time_end >= bucket["max"]]) < min_size
            ):
                # Stop when we reach the end.
                if bucket["max"] > max_time:
                    break
                bucket["max"] += 1

            # Create the next bucket at the previous max time, incremented by 1 timestamp.
            next_time = bucket["max"] + 1
            buckets[len(buckets)] = {"min": next_time, "max": next_time}

        # Create a map from each year to its buckets
        year_map = {}
        for idx, times in buckets.items():
            min, max = times["min"], times["max"]
            for time in range(min, max + 1):
                year_map[time] = idx

        # For some reason this gives a `SettingWithCopyWarning`.
        # It seems to work fine apart from the console spam.
        triples.update(triples.time_begin.map(year_map))
        triples.update(triples.time_end.map(year_map))

        return triples, buckets

    @staticmethod
    def aggregate_time_per_predicate(triples: pd.DataFrame, min_size: int) -> pd.DataFrame:
        """ Aggregate fact timestamps into timespans s.t. that there is a minimum number of facts
            for each predicate, for each timespan. """
        relations = [x for x in triples.relation_id.unique()]

        translated = pd.DataFrame(columns=triples.columns)
        for rela in relations:
            subset = triples.loc[triples.relation_id == rela]
            aggregated, _ = Utils.aggregate_years(subset, min_size)
            translated = translated.append(aggregated)

        return translated

    @staticmethod
    def filter_dict_from_set(to_filter: dict, set: set) -> dict:
        to_remove = []
        for key in to_filter.keys():
            if key not in set:
                to_remove.append(key)

        for key in to_remove:
            del to_filter[key]

        return to_filter

    @staticmethod
    def get_group_idx(groups, init_value: Any, compare_func: Callable[[int, Any, Any], Tuple[bool, Any]],) -> int:
        """ groups: groups to select from
            init_value: value to initialise the search with
            compare_func: function that takes the currently evaluated group idx, the current best value and the groups, 
                and returns (whether it is better, the score of the current)
        """
        group_idx = -1
        best_value = init_value

        for idx in groups.groups.keys():
            better, curr_value = compare_func(idx, best_value, groups)
            if better:
                group_idx, group_size = idx, curr_value

        return group_idx

    @staticmethod
    def get_largest_group_idx(groups, min_size: int = 0) -> int:
        """ Given a pandas DataFrameGroupBy object, return the index of the group with the largest
            number of entries, with at least min_size members. If none is found -1 is returned."""
        return Utils.get_group_idx(
            groups,
            min_size,
            lambda curr_idx, best_value, groups: (
                len(groups.get_group(curr_idx)) > best_value,
                len(groups.get_group(curr_idx)),
            ),
        )

    @staticmethod
    def get_smallest_group_idx(groups, min_size: int) -> int:
        """ Given a pandas DataFrameGroupBy object, return the index of the group with the smallest
            number of entries of at least min_size. If none is found -1 is returned."""
        return Utils.get_group_idx(
            groups,
            999999999,
            lambda curr_idx, best_value, groups: (
                len(groups.get_group(curr_idx)) <= best_value and len(groups.get_group(curr_idx)) >= min_size,
                len(groups.get_group(curr_idx)),
            ),
        )

    @staticmethod
    def pretty_print(row, entities: dict, relations: dict):
        subject = entities[row.subject_id]
        relation = relations[row.relation_id]
        obj = entities[row.object_id]

        print(f"({subject},{relation},{obj})[{row.time_begin}-{row.time_end}]")

    @staticmethod
    def print_data_properties(triples: pd.DataFrame) -> None:
        print("# entities: " + str(len(triples.subject_id.append(triples.object_id).unique())))
        print("# relations: " + str(len(triples.relation_id.unique())))
        print("# triples: " + str(len(triples)))

    @staticmethod
    def sanity_check(triples: pd.DataFrame, entities: dict, relations: dict) -> None:
        # Are there any entities which do not occur?
        empty_entities = [x for x, y in (Utils.get_entity_counts(triples, entities).items()) if y == 0]
        print(str(len(empty_entities)) + " entities do not occur in the dataset.")
        if len(empty_entities) > 0:
            print("These are: " + ",".join([str(x) for x in empty_entities]))

        # Are there any relations which do not occur?
        empty_relations = [
            x for x, y in (Utils.get_group_counts(triples.groupby("relation_id"), relations).items()) if y == 0
        ]
        print(str(len(empty_relations)) + " relations do not occur in the dataset.")
        if len(empty_relations) > 0:
            print("These are: " + ",".join([str(x) for x in empty_relations]))

        # Are there any timeranges that are invalid?
        time_diffs = triples["time_end"] - triples["time_begin"]
        invalid_times = time_diffs[time_diffs < 0].index
        print("Found " + str(len(invalid_times)) + " triples with invalid scopes")

        if len(invalid_times) > 0:
            joined_indices = ", ".join(str(x) for x in invalid_times)
            print("Indices: " + joined_indices)

    @staticmethod
    def print_time_info(triples: pd.DataFrame) -> None:
        # Print average fact length
        time_diffs = triples["time_end"] - triples["time_begin"]
        print("Mean timespan: " + str(time_diffs.mean()))
        print("Largest timespan: " + str(time_diffs.max()))
        print("Smallest timespan: " + str(time_diffs.min()))

    @staticmethod
    def scope_name(current_name: str, begin: float, end: float) -> str:
        # Handling of floor/ceil should be the same as in split_groups
        return current_name + "[" + str(math.ceil(begin)) + "-" + str(math.floor(end)) + "]"

    @staticmethod
    def get_group_counts(grouped_triples, items: dict) -> dict:
        counts = {x: 0 for x in items.keys()}
        for idx, size in (grouped_triples.size()).iteritems():
            counts[idx] = size

        return counts

    @staticmethod
    def get_group_safe(grouped_triples, group_id: int):
        """ Attempt to get a group with the given key.
            If the group does not exist, an empty dataframe with the same columns is returned."""
        try:
            return grouped_triples.get_group(group_id)
        except KeyError:
            return pd.DataFrame(columns=grouped_triples.first().columns)

    @staticmethod
    def get_entity_counts(triples: pd.DataFrame, entities: dict) -> dict:
        obj_counts = triples.subject_id.value_counts()
        subj_counts = triples.object_id.value_counts()

        return {x: obj_counts.get(x, 0) + subj_counts.get(x, 0) for x in entities.keys()}

    @staticmethod
    def get_entity_timerange(triples: pd.DataFrame) -> Dict[Id, Dict[Id, Time]]:
        """Get a dictionary containing the first and last occurrence of an entity in the dataset."""
        # Group entities and get min/max year for each subject and object seperately.
        subject_spans = (
            triples.rename(columns={"subject_id": "id"}).groupby("id").agg({"time_begin": "min", "time_end": "max"})
        )
        object_spans = (
            triples.rename(columns={"object_id": "id"}).groupby("id").agg({"time_begin": "min", "time_end": "max"})
        )

        # Merge the subjects and object results, get the min/max from both results.
        merged = pd.merge(subject_spans, object_spans, how="outer", on="id")
        time_begin = merged[["time_begin_x", "time_begin_y"]].min(axis=1)
        time_end = merged[["time_end_x", "time_end_y"]].max(axis=1)

        # Recombine them into a df, and then export it as a dictionary.
        return (
            pd.DataFrame(time_begin, columns=["time_begin"], dtype="int32")
            .join(pd.DataFrame(time_end, columns=["time_end"], dtype="int32"))
            .to_dict(orient="index")
        )

    @staticmethod
    def get_relation_timerange_all(triples: pd.DataFrame):
        """Get a dictionary containing the first and last occurrence of each relation in the dataset."""
        return triples.groupby("relation_id").agg({"time_begin": "min", "time_end": "max"}).to_dict(orient="index")

    @staticmethod
    def get_relation_timerange(triples: pd.DataFrame, pred: int) -> Tuple[int, int]:
        """ Get the (first, last) occurrences of the given predicate in the given triples. """
        slice = triples[triples.relation_id == pred]
        return slice.time_begin.min(), slice.time_end.max()

    @staticmethod
    def parse_helper(name: str, default: Any, parser: Callable) -> Any:
        value = input(name + " (default '" + str(default) + "'): ")
        if value == "":
            return default
        else:
            parsed = parser(value)
            if parsed == ValueError:
                sys.exit("Invalid Argument")
            return parsed

    @staticmethod
    def df_from_name_dict(items: dict) -> pd.DataFrame:
        """ Convert an entity or relationship dictionary (key => name) to a dataframe.
            ID's become a index from 1,2,...,n without any gaps."""
        out = pd.DataFrame.from_dict(items, orient="index", columns=["name"])
        return out.reset_index().drop(["index"], axis=1)

    @staticmethod
    def roll_sum(idx: int, values: pd.Series):
        roll_sum = values.rolling(2).sum()

        end_timestamp, val = roll_sum.idxmin(), roll_sum.min()
        start_timestamp = values.index[np.searchsorted(values.index, end_timestamp) - 1]

        return idx, val, start_timestamp, end_timestamp

    @staticmethod
    def normalize(vec: np.array):
        norm = np.linalg.norm(vec, 2)
        if norm <= 0.00001:  # epsilon
            return vec
        return vec / norm

