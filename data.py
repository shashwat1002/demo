import datasets
from typing import *
import pyarrow as pa

# First make a config


class CustomDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for CustomDataset"""

    def __init__(self, features, files, labels_classes, **kwargs):
        """BuilderConfig for CustomDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CustomDatasetConfig, self).__init__(**kwargs)

        self.features = features
        self.label_classes = labels_classes
        self.files = files


class CustomDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CustomDatasetConfig(
            features=datasets.Features(
                {
                    "op1": datasets.Value("int32"),
                    # "guess": Dataset.Value("string"),
                    "op1": datasets.Value("int32"),
                }
            ),
            labels_classes=datasets.Features(
                {
                    "result": datasets.Value("int32"),
                }
            ),
            files=["train.txt", "val.txt", "test.txt"],
        )
    ]


    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "op1": datasets.Value("int32"),
                    "op2": datasets.Value("int32"),
                    "result": datasets.Value("int32"),
                }
            ),
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.files[0]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": self.config.files[1]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": self.config.files[2]}
            ),
        ]

    def _generate_examples(self, filepath):
        print(filepath)
        with open(filepath) as f:
            for id, row in enumerate(f):
                row = row.split(" ")
                row = [int(x) for x in row]
                # print(id, row[0])

                yield id, {
                    "op1": row[0],
                    "op2": row[1],
                    "result": row[0] + row[1],
                }
