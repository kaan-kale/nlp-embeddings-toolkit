"""DOCSTRING.""" #TODO: Write docstring.

from datasets import load_dataset


class GlueDatasetLoader:
    """Load the GLUE dataset from the HuggingFace Datasets library."""
    def load_glue_dataset(self, task_name, split="train"):
        """Load the GLUE dataset from the HuggingFace Datasets library.

        Args:
            task_name (str): GLUE task name.
            split (str, optional): Split of the dataset. Defaults to "train".

        Returns:
            dataset: The GLUE dataset.
        """
        return load_dataset("glue", task_name)[split]

