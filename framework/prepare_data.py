import argparse
import sys

from dataloaders.data import Data
from dataloaders.loader import Dataloader
from transformations.transformations import Transform
from transformations.loader import TransformLoader


if __name__ == "__main__":
    data_loader = Dataloader()
    transform_loader = TransformLoader()

    # Parse arguments.
    parser = argparse.ArgumentParser(description="Prepare data script")
    parser.add_argument(
        "dataset_name", type=str, help="Name of the dataset to be imported", choices=data_loader.get_datasets(),
    )
    parser.add_argument(
        "transformation", type=str, help="Type of transformation to be applied", choices=transform_loader.get_t_types(),
    )

    parser.add_argument(
        "apply_to", type=str, help="Apply to which components", choices=["entities", "relations"],
    )

    parser.add_argument(
        "filterdupes",
        type=str,
        help="Whether to remove duplicates triples from the dataset",
        choices=["inter", "intra", "both", "none"],
        default="none",
    )

    parser.add_argument(
        "--nonames", action="store_true", help="Skip conversion to names from ID's", default=False,
    )

    args = parser.parse_args()

    parsed_options = {
        "filter_dupes": args.filterdupes,
        "nonames": args.nonames,
    }

    dataset_name = args.dataset_name.casefold()
    dataset = data_loader.load(dataset_name, parsed_options)

    print("Loaded data.")

    # Transform data.
    transformation = args.transformation.strip().casefold()
    target = args.apply_to.strip().casefold() if hasattr(args, "apply_to") else ""

    transformer = transform_loader.load(transformation, target, dataset)

    transformer.transform()
    print("Transformed data.")

    # Output data. Special path for some values to make it easier to generate multiple examples.
    path = transformer.get_dest_path()

    if transformer.EPSILON_FACTOR > 0:
        path += "-epsilon-" + str(transformer.EPSILON_FACTOR)
    elif transformer.GROW_FACTOR > 0:
        path += "-grow-" + str(transformer.GROW_FACTOR)
    elif transformer.SHRINK_FACTOR > 0:
        path += "-shrink-" + str(transformer.SHRINK_FACTOR)

    path += "-" + dataset.options["filter_dupes"]
    dataset.output(path)

    # Output an about.txt
    with open(path + "/about.txt", "w") as file:
        file.write(f"# triples: {len(dataset.triples)} \n")
        file.write(f"# entities: {len(dataset.entities)} \n")
        file.write(f"# relations: {len(dataset.relations)} \n")
        file.write(f"# timesteps: {dataset.triples['time_begin'].append(dataset.triples['time_end']).nunique()} \n")

        file.write(f"# test triples: {len(dataset.triples[dataset.triples.source == 'test'])} \n")
        file.write(f"# valid triples: {len(dataset.triples[dataset.triples.source == 'valid'])} \n")
        file.write(f"# train triples: {len(dataset.triples[dataset.triples.source == 'train'])} \n")

        file.write(f"Measure method:  {transformer.MEASURE_METHOD}  \n")
        file.write(f"Target Size :  {transformer.TARGET_SIZE}  \n")
        file.write(f"Grow Factor:  {transformer.GROW_FACTOR}  \n")
        file.write(f"Shrink Factor:  {transformer.SHRINK_FACTOR}  \n")
        file.write(f"Epsilon Factor: {transformer.EPSILON_FACTOR}  \n")
        file.write(f"Search method: {transformer.SEARCH_METHOD}  \n")

        for option, value in parsed_options.items():
            file.write(str(option) + ": " + str(value) + "\n")

    print("Done!")
