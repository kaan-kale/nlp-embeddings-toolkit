"""DOCSTRING."""  # TODO: Write docstring.

import os
import torch


class ResultsSaver:
    """DOCSTRING."""  # TODO: Write docstring.
    @staticmethod
    def save_results(self, results, save_path):
        """DOCSTRING."""  # TODO: Write docstring.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        outputs_save_path = os.path.join(save_path, "outputs.pt")
        labels_save_path = os.path.join(save_path, "labels.pt")
        torch.save(results["outputs"], outputs_save_path)
        torch.save(results["labels"], labels_save_path)