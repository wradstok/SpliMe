#!/usr/bin/python3
import os
import numpy as np
import argparse
import pandas as pd
import math
import subprocess
import sys

from typing import Tuple, Dict
from ampligraph.datasets import load_from_csv
from ampligraph.latent_features import TransE, ComplEx
from ampligraph.evaluation import (
    evaluate_performance,
    mr_score,
    mrr_score,
    hits_at_n_score,
)
from ampligraph.utils import save_model, restore_model


def get_args():
    parser = argparse.ArgumentParser(description="Prepare data script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to be imported")
    parser.add_argument("model_name", type=str, help="Name of the model to be used")
    parser.add_argument("batch_size", type=int, help="Size of batches to shoot for")
    parser.add_argument(
        "--localrun", dest="local_run", default=False, action="store_true", help="Fix path for debugging in vscode",
    )
    parser.add_argument(
        "--test_only",
        dest="test_only",
        default=False,
        action="store_true",
        help="Skip training and instead load existing model",
    )

    parser.add_argument(
        "--emb_size", type=int, dest="emb_size", nargs="?", help="Dimensionality parameter ", default=100,
    )

    parser.add_argument(
        "--scale",
        dest="scale",
        default=False,
        action="store_true",
        help="whether to scale the # parameters to # of splits.",
    )
    return parser.parse_args()


def init_model(model_name: str, batch_count: int, dataset: str, preds: int, emb_size: int, scale: bool):
    if model_name == "TransE-params":
        # Modify emebdding dimension s.t. emb_size per original predicate is achieved.
        params = 1
        if scale:
            if "wiki" in dataset:
                params = 24 / preds
            elif "yago" in dataset:
                params = 10 / preds
            elif "icews14" in dataset:
                params = 230 / preds
            else:
                raise ValueError()

        params = int(emb_size * params)
        print(f"Set embedding size to {params}")
        return TransE(
            k=params,
            optimizer="adam",
            batches_count=batch_count,
            optimizer_params={"lr": 0.001},
            loss="self_adversarial",
            eta=500,
            epochs=200,
            verbose=True,
        )
    if model_name == "TransE-final":
        return TransE(
            k=emb_size,
            optimizer="adam",
            batches_count=batch_count,
            optimizer_params={"lr": 0.001},
            loss="self_adversarial",
            eta=500,
            epochs=200,
            verbose=True,
        )
    else:
        raise ValueError


def get_model(model_name: str, params: dict):
    # Set model instance
    if model_name == "TransE":
        if len(params) > 0:
            return TransE(
                k=params["k"], eta=params["eta"], loss=params["loss"], epochs=300, batches_count=params["batches_count"]
            )
        else:
            return TransE
    elif model_name == "ComplEx":
        if len(params) > 0:
            return ComplEx(
                k=params["k"],
                eta=params["eta"],
                loss=params["loss"],
                epochs=300,
                batches_count=params["batches_count"],
                regularizer="LP",
            )
        else:
            return ComplEx
    else:
        raise ValueError


def load_dataset(
    folder: str, removed: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    train: pd.Dataframe = load_from_csv(folder, "train.txt", sep="\t")
    test: pd.Dataframe = load_from_csv(folder, "test.txt", sep="\t")
    valid: pd.Dataframe = load_from_csv(folder, "valid.txt", sep="\t")

    test_size, valid_size = len(test), len(valid)

    # Test and valid split can contain entities that do not occur in the training set.
    #  Let's filter these triple so we can actually run something.
    train_entities = np.unique(np.append(train[:, 0], train[:, 2]))
    valid_entities = np.array([x for x in np.unique(np.append(valid[:, 0], valid[:, 2])) if x in train_entities])
    test_entities = np.array([x for x in np.unique(np.append(test[:, 0], test[:, 2])) if x in train_entities])

    valid = np.array([np.array([s, p, o]) for s, p, o in valid if s in valid_entities and o in valid_entities])
    test = np.array([np.array([s, p, o]) for s, p, o in test if s in test_entities and o in test_entities])

    # Now lets do the same for predicates, because the test/valid sets may contain predicates with a temporal
    # scope that is outside of the scope of the train set.
    train_predicates = np.unique(train[:, 1])
    valid_predicates = np.array([x for x in valid[:, 1] if x in train_predicates])
    test_predicates = np.array([x for x in test[:, 1] if x in train_predicates])

    valid = np.array([[s, p, o] for s, p, o in valid if p in valid_predicates])
    test = np.array([[s, p, o] for s, p, o in test if p in test_predicates])

    removed_test, removed_valid = test_size - len(test), valid_size - len(valid)

    removed["test"] += removed_test
    removed["valid"] += removed_valid

    print("Removed " + str(removed_test) + " from test, " + str(len(test)) + " remaining")
    print("Removed " + str(removed_valid) + " from valid, " + str(len(valid)) + " remaining")

    return train, test, valid, removed


def train_and_evaluate(model, train, test, valid, save_path: str) -> Dict[str, float]:
    model.fit(train, early_stopping=False, early_stopping_params={"x_valid": valid, "criteria": "mrr"})

    save_model(model, save_path)

    return evaluate(model, train, test, valid)


def evaluate(model, train, test, valid) -> Dict[str, float]:
    ranks = evaluate_performance(
        test,
        model=model,
        filter_triples=np.concatenate((train, valid, test)),
        corrupt_side="s,o",
        filter_unseen=False,
        verbose=True,
    )

    result = {
        "mrr": mrr_score(ranks),
        "mr": mr_score(ranks),
        "hits_1": hits_at_n_score(ranks, n=1),
        "hits_3": hits_at_n_score(ranks, n=3),
        "hits_10": hits_at_n_score(ranks, n=10),
    }

    return result


def get_path(curr_path: str, index: int) -> str:
    return curr_path


if __name__ == "__main__":
    print("Initializing")

    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "tensorflow==1.15.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "ampligraph==1.3.1"])

    args = get_args()
    dataset_name = args.dataset_name.strip()
    model_name = args.model_name.strip()

    # Handle path.
    base_path = str(os.getcwd()) + "/KnowledgeGraphEmbedding/" if args.local_run else str(os.getcwd())
    data_path = base_path + "/transformed_data/" + dataset_name + "/"
    model_location = base_path + "/models/" + model_name + "-" + dataset_name

    datasets = []
    removed = {"test": 0, "valid": 0}
    
    train, test, valid, removed = load_dataset(data_path, removed)
    datasets.append((train, test, valid))

    results = []
    for i, (train, test, valid) in enumerate(datasets):
        curr_path = get_path(model_location, i)
        if args.test_only:
            print("Restoring model")
            model = restore_model(curr_path)
            results.append(evaluate(model, train, test, valid))
        else:
            print("Training model")
            batch_count = int(math.ceil(len(train) / args.batch_size))
            # Pass in the number of unique predicates
            model = init_model(
                model_name, batch_count, dataset_name, len(np.unique(train[:, 1])), args.emb_size, args.scale
            )
            
            results.append(train_and_evaluate(model, train, test, valid, curr_path))

    # Calculate average result.
    total_result = {}
    for result in results:
        for metric, value in result.items():
            total_result[metric] = total_result.get(metric, 0) + value

    print("Summary:")
    for metric, value in total_result.items():
        print(metric + ": " + str(value / len(results)))

