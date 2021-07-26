# Instructions

Thank you for taking an interest in our work. The following are instruction on how to reproduce the results displayed in our paper. 

To get started, you must have 3.6 <= Python (64 bit) >= 3.7 installed. Older or newer versions unfortunately do not work due to dependency conflicts. Next, you can install the required dependencies by running the following in your shell.

> py -m pip install -r requirements.txt

## Structure
The `framework` directory contains the SpliMe code to load the different datasets (Wikidata12k, YAGO11k, ICEWS14) under the `dataloaders` subdirectory. The `transformations` subdirectory contains the different SpliMe transformations that can be applied. 

The `source_data` directory contains the datasets on which SpliMe has been evaluated under a subdirectory indicating the paper the data was taken from. 

The `KnowledgeGraphEmbedding` directory contains the transformed datasets obtained by applying SpliMe in the `transformed_data` subdirectory, as well as the models/embeddings learned in the `models` subdirectory.

## Applying SpliMe
To run SpliMe, first switch into the `framework` directory, then call `prepare_data.py` with the required arguments:

> py prepare_data.py {dataset} {SpliMe method} {entities/predicates} {filtertype}

Any hyperparameters required for the specific SpliMe approach will be asked for afterwards in an interactive manner. Please note that not all SpliMe approaches have been implemented for entities. For instance, we do not have a CPD-based splitting implementation for entities.

The following table is a list of SpliMe method names and their corresponding argument call:

Paper| Code
-------|--------
Vanilla baseline | vanilla
Random baseline | baseline
Timestamping     | timestamp
Parameterized splitting | split
CPD-based splitting     | prox
Merging | merge

To give a further example, the transformed dataset for our best Wikidata12k result was obtained using merge and can be generated with the following commands:

> py prepare_data.py wikidata12k merge relations inter
> 4

The first command here calls the correct SpliMe function, the second command tells it the shrink factor (hyperparameter) we wish to apply. SpliMe will now start transforming the dataset. The result will be output in `KnowledgeGraphEmbedding/transformed_data` under the folder `wiki_data_merge_rel-shrink-4.0-inter`. 

**Note:** calculating signature vectors for the CPD-based efficient splitting method take a while the first time it is run a dataset. However, the results are cached so that subsequent runs will be significantly faster.

## Training and evaluation
Once SpliMe has been applied to a dataset to transform it, we use the Ampligraph library to actually perform knowledge graph embedding. To do this, first navigate to the `KnowledgeGraphEmbedding` directory then run:

> py run_ampli.py {transformed_data_name} TransE-final 500

For instance, to train & evaluate the optimal Wikidata12k dataset we generated earlier, run:

> py run_ampli.py wiki_data_merge_rel-shrink-4.0-inter TransE-final 500

This will start training a TransE model for the given dataset. When training has been completed, the model is evaluated using the test set. The result of this will be output to the console. Note, the training process may take several hours on a CPU.