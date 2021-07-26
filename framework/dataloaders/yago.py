import numpy as np
import pandas as pd

from dataloaders.data import Data


class LoadYago11kData(Data):
    """ Load Yago11k Dataset"""

    def __init__(self, options: dict):
        super().__init__(options)
        path = super().get_source_path()

        entities = pd.read_table(
            path + "/entity2id.txt", header=None, names=["name", "id", "occurs_since", "occurs_until"], index_col="id",
        )
        self.entities = entities["name"].to_dict()

        relations = pd.read_table(path + "/relation2id.txt", header=None, names=["name", "id"], index_col="id")
        self.relations = relations["name"].to_dict()

        self.triples = self.get_data_with_source(
            path, ["subject_id", "relation_id", "object_id", "time_begin", "time_end"]
        )

        self.triples = pd.DataFrame(
            data=np.apply_along_axis(self.fix_years, 1, self.triples.values), columns=self.triples.columns,
        )

        self.check_self()

        self.prepare()

    def get_name(self) -> str:
        return "yago"

    def from_paper(self) -> str:
        return "hyte"

    def fix_years(self, numpy_row) -> np.ndarray:
        begin = self.try_parse_year(numpy_row[3])
        end = self.try_parse_year(numpy_row[4])

        if end == "weird":
            end = 2017
        numpy_row[3], numpy_row[4] = int(begin), int(end)
        return numpy_row