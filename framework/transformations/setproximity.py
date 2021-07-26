from functools import partial
from types import FunctionType
from typing import Dict, List, Set
from networkx.algorithms.simple_paths import all_simple_paths

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ruptures as rpt
import pickle as pcl
import math
import networkx as nx

from dataloaders.data import Data
from my_types import Id, Name, Time
from utils import Utils
from pathlib import Path

from transformations.transformations import Transform, TransformTarget


class SetProximity(TransformTarget):
    def __init__(self):
        super().__init__()

    def get_entity_transform(self, data: Data) -> Transform:
        return NotImplementedError()

    def get_relation_transform(self, data: Data) -> Transform:
        return Relations(data)


class EntPair:
    """ Helper class to store an entity pair. Subj/obj order does not matter. So EntPair(X, Y) == EntPair(Y, X). 
        Due to implementing comparison operators based on hash, these pairs can be sorted. 
        The ordering is abritary, but consistent. """

    def __init__(self, a: int, b: int):
        self.a, self.b = a, b
        self.hash = frozenset([a, b]).__hash__()

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, o: object) -> bool:
        return self.hash == o.__hash__()

    def __lt__(self, o: object) -> bool:
        return self.hash < o.__hash__()

    def __le__(self, o: object) -> bool:
        return self.hash <= o.__hash__()

    def __gt__(self, o: object) -> bool:
        return self.hash > o.__hash__()

    def __ge__(self, o: object) -> bool:
        return self.hash >= o.__hash__()


class Signature:
    """ Signature of a predicate at a given timestamp. """

    def __init__(self) -> None:
        self.pairs: Dict[EntPair, float] = {}

    def add_pair(self, pair: EntPair, score: float) -> None:
        self.pairs[pair] = score

    def exists(self, pair):
        return pair in self.pairs

    def prepare(self) -> List[float]:
        order = sorted(self.pairs)
        return [self.pairs[o] for o in order]


class Signatures:
    """ For a single predicate, store its signature at every timestamp. """

    def __init__(self) -> None:
        self.pairs: Dict[Time, Signature] = {}
        self.all_pairs: Set[EntPair] = set()

    def add_pair(self, time: int, pair: EntPair, score: float) -> None:
        if time not in self.pairs:
            self.pairs[time] = Signature()

        self.pairs[time].add_pair(pair, score)
        self.all_pairs.add(pair)

    def exists(self, time: int, pair: EntPair) -> bool:
        return time in self.pairs and self.pairs[time].exists(pair)

    def prepare(self) -> Dict[Time, np.array]:
        """Export the signature as dictionary, containing for each timestamp a list of scores.
            The indices of these list correspond to a specific EntPair and are equal over all timestamps. """
        result_dict: Dict[Time, List[float]] = {time: [] for time in self.pairs.keys()}

        for time, _ in self.pairs.items():
            # First add any any missing elements to each set with the score set to a neutral 0.
            for pair in self.all_pairs:
                if not self.pairs[time].exists(pair):
                    self.pairs[time].add_pair(pair, 0)

            # Then turn it into a sorted list
            result_dict[time] = np.array(self.pairs[time].prepare())

        return result_dict


class Dist_Helper:
    def __init__(self) -> None:
        # Save node_neighbourhoods to prevent excess computations.
        self.nn: Dict[Time, Dict[Id, Set]] = {}
        self.metrics = {
            "jacard": partial(Dist_Helper.jacard),
            "pref": partial(Dist_Helper.pref_attach),
            "adar": partial(Dist_Helper.adar),
            "katz": partial(Dist_Helper.weighted_katz),
        }

        # Only used in case of katz
        # Stores adjacency matrices for a pred => time -> power
        self.graphs: Dict[Id, Dict[Time, nx.Graph]] = {}

    def set_cutoff(self, cutoff: int):
        self.CUTOFF = cutoff

    def get_metrics(self) -> Dict[str, FunctionType]:
        return self.metrics

    def get_metric(self, name: str) -> FunctionType:
        return self.metrics[name]

    def slice_time(self, slice: pd.DataFrame, time: int) -> pd.DataFrame:
        return slice[(slice.time_begin <= time) & (time <= slice.time_end)]

    def node_neighbourhood(self, triples: pd.DataFrame, time: int, ent: int) -> Set[int]:
        """ Get the node neighbourhood of the given entity, only using predicate links of the given type.
            Ignores whether the entity occurs as a subject or object in the relation (i.e. an undirected graph).
            Note: the dataframe should be pre-sliced to only contain facts for the relevant predicate."""
        # Create entry for the given time if it does not yet exist.
        if not time in self.nn:
            self.nn[time] = {}

        # Create entry for the (time, ent) combination if it does not yet exist.
        if not ent in self.nn[time]:
            triples = self.slice_time(triples, time)
            triples = triples[(triples.subject_id == ent) | (triples.object_id == ent)]
            self.nn[time][ent] = set(triples.subject_id.append(triples.object_id).unique())

        return self.nn[time][ent]

    def jacard(self, slice: pd.DataFrame, time: int, pair: frozenset) -> float:
        """ Calculate the jacard distance. """
        x = Dist_Helper.node_neighbourhood(self, slice, time, pair.a)
        y = Dist_Helper.node_neighbourhood(self, slice, time, pair.b)

        return len(x.intersection(y)) / len(x.union(y))

    def pref_attach(self, slice: pd.DataFrame, time: int, pair: EntPair) -> float:
        """ Preferential attachment. """
        x = Dist_Helper.node_neighbourhood(self, slice, time, pair.a)
        y = Dist_Helper.node_neighbourhood(self, slice, time, pair.b)
        return len(x) * len(y)

    def adar(self, slice: pd.DataFrame, time: int, pair: EntPair) -> float:
        x = Dist_Helper.node_neighbourhood(self, slice, time, pair.a)
        y = Dist_Helper.node_neighbourhood(self, slice, time, pair.b)

        # Normally, the node_neighbourhood of each node in the intersection of x,y contains at least 2 nodes: x and y
        # However, it is possible for an entity to be in a relationship with itself. In this case, x,y are equal.
        # If the node is not associated with any other nodes than its node_neighbourhood is of length 1.
        # Now, log(1) == 0, which leads to a division by zero error.
        # To prevent this, we check whether the nodes are equal and in that case simply return a score of 0.
        if x == y:
            return 0

        return sum(
            map(lambda z: 1 / math.log10(len(Dist_Helper.node_neighbourhood(self, slice, time, z))), x.intersection(y))
        )

    def weighted_katz(self, slice: pd.DataFrame, time: int, pair: EntPair) -> float:
        slice = self.slice_time(slice, time)

        # Create the graph for this (predicate, time) pair if we do not have it yet.
        pred = slice.iloc[0].relation_id

        if pred not in self.graphs:
            self.graphs[pred] = {}

        if time not in self.graphs[pred]:
            graph = nx.Graph()
            uniques = slice.value_counts(["subject_id", "relation_id", "object_id"])
            for ((sub, _, obj), count) in uniques.iteritems():
                graph.add_edge(sub, obj, weight=count)

            self.graphs[pred][time] = graph

        # Get the adjacency matrix
        graph = self.graphs[pred][time]

        beta = 0.005

        # Collect all paths and gather them by length.
        path_counts = {}
        for path in nx.all_simple_paths(graph, pair.a, pair.b, cutoff=self.CUTOFF):
            length = len(path) - 1  # Because it counts a direct path as 2
            path_counts[length] = path_counts.get(length, 0) + 1

        # Weighted katz, so # paths with length 1 is the number of times the pair appears.
        path_counts[1] = len(
            slice[
                ((slice.subject_id == pair.a) & (slice.object_id == pair.b))
                | (slice.subject_id == pair.b) & (slice.object_id == pair.a)
            ]
        )

        score = 0
        for length, count in path_counts.items():
            score += beta ** length * count

        return score


class Relations(Transform):
    """ Add temporal scope by adding splits to relations"""

    def __init__(self, data: Data):
        # First load the possible options from the distance helper, then initialize the rest.
        self.dist_helper = Dist_Helper()

        super().__init__(data)
        self.init_parameters()

        self.data.triples, self.data.buckets = Utils.aggregate_years(self.data.triples)

        # Calculating signatures is slow.. So we save the result using pickle and then re-load it.
        path = Path(__file__).parent.parent.absolute()
        name = (
            self.data.get_name() + "-" + self.DISTANCE_METRIC + ".p"
        )
        target = path.joinpath("cache", name)

        # Check if signature has been calculated before.
        if target.exists():
            print("Loading from pre-calculated signature data")
            self.pred_signatures = pcl.load(open(str(target), "rb"))
        else:
            # Calculate the signature of each set.
            self.pred_signatures: Dict[int, Signatures] = {}
            for i, pred in enumerate(self.data.relations.keys()):
                if i % 5 == 0:
                    print(f"Calculated signatures for {i}/{len(self.data.relations.keys())} predicates")
                self.pred_signatures[pred] = self.calc_signatures(pred)

            pcl.dump(self.pred_signatures, open(str(target), "wb"))

        # Store the first & last occurences of each predicate
        self.pred_timerange = Utils.get_relation_timerange_all(self.data.triples)

    def calc_split_points(self, signatures: Signatures) -> List[int]:
        """ Perform CPD on a single predicate using the given signatures. """
        # There can be no splits if there are not at least two elements.
        if len(signatures.pairs) <= 1:
            return []

        ready = signatures.prepare()
        ready_ordered = np.array([ready[x] for x in sorted(ready.keys())])

        # Normalize vectors before performing CPD.
        # This prevents issues with epsilon behaving differently for different datasets.
        normalized = np.array(list(map(Utils.normalize, ready_ordered)))
        model = rpt.BottomUp(model="rbf", jump=1, min_size=1).fit(normalized)
        split_points = model.predict(epsilon=self.EPSILON_FACTOR)[:-1]

        # Split points are returned as indices in the original list, but we need to find
        # which actual times they correspond to, as there can be gaps.
        # E.g. index 20 in ready could correspond to time 35.
        return [sorted(ready.keys())[x] for x in split_points]

    def calc_signatures(self, pred: int) -> Signatures:
        """ Calculate the signature vector of the given predicate."""
        signatures = Signatures()
        slice = self.data.triples[self.data.triples.relation_id == pred]

        for row in slice.itertuples():
            # Create the relevant entity pair.
            pair = EntPair(row.subject_id, row.object_id)

            # Calculate its score for every time in which it exists, if we have not done so previously.
            for time in range(row.time_begin, row.time_end + 1):
                if not signatures.exists(time, pair):
                    score = self.dist_helper.get_metric(self.DISTANCE_METRIC)(self.dist_helper, slice, time, pair)
                    signatures.add_pair(time, pair, score)

        return signatures

    def get_name(self) -> str:
        return "rel_dist_" + self.DISTANCE_METRIC

    def init_parameters(self) -> None:
        self.EPSILON_FACTOR = Utils.parse_helper("Epsilon", 10, float)
        self.DISTANCE_METRIC = Utils.parse_helper(
            "Distance metric",
            "jacard",
            lambda metric: metric if metric in self.dist_helper.get_metrics() else ValueError,
        )

        if self.DISTANCE_METRIC == "katz":
            self.dist_helper.set_cutoff(Utils.parse_helper("Cutoff", 10, int))

    def generate_new_pred(self, pred: int, prev_split: int, split: int) -> None:
        """Generate a new predicate for given predicate id."""
        new_pred_id = self.get_next_relation()
        self.pred_names[new_pred_id] = Utils.scope_name(self.data.relations[pred], prev_split, split)
        self.pred_time_map[pred][split] = new_pred_id

    def transform(self) -> None:
        # Calculate the split indices for each predicate.
        print("Calculating split points.")
        pred_splits: Dict[Id : List[int]] = {}
        for pred, signatures in self.pred_signatures.items():
            pred_splits[pred] = self.calc_split_points(signatures)
            # Add one more split all the way at the end.
            pred_splits[pred].append(self.pred_timerange[pred]["time_end"] + 1)

        # Generate a new predicate for each split interval.
        self.pred_time_map: Dict[Id, Dict[Time, Id]] = {x: {} for x in self.data.relations.keys()}
        self.pred_names: Dict[Id, Name] = {}

        for pred, splits in pred_splits.items():
            # Prev split starts at the first timestamp associated with this predicate.
            prev_split = self.data.triples[self.data.triples.relation_id == pred].time_begin.min()

            for split in splits:
                # Generate new predicate.
                self.generate_new_pred(pred, prev_split, split)
                prev_split = split

            # Also generate one for the very end. In the case of no splits, this is the only one.
            self.generate_new_pred(pred, prev_split, self.pred_timerange[pred]["time_end"] + 1)

        # Perform splitting
        new_triples = []
        print("Performing transformation.")
        for row in self.data.triples.itertuples():
            orig_pred = row.relation_id
            split_times = pred_splits[orig_pred]

            first_idx = np.searchsorted(split_times, row.time_begin, "right")
            last_idx = np.searchsorted(split_times, row.time_end, "right")

            splits_in_range = split_times[first_idx : last_idx + 1]

            prev_split = split_times[np.searchsorted(split_times, splits_in_range[0], "left") - 1]
            for split in splits_in_range:
                new_triples.append(
                    [row.subject_id, self.pred_time_map[orig_pred][split], row.object_id, prev_split, split, row.source]
                )
                prev_split = split

        # Save converted data.
        self.data.triples = pd.DataFrame(
            new_triples, columns=["subject_id", "relation_id", "object_id", "time_begin", "time_end", "source",],
        )
        self.data.relations = self.pred_names

