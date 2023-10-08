"""DOCSTRING."""  # TODO: Write docstring.

import torch

from tqdm import tqdm

class EmbeddingExtractor:
    """DOCSTRING."""  # TODO: Write docstring.
    def __init__(self, model, tokenizer, device=None):
        """DOCSTRING."""  # TODO: Write docstring.
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

    def preprocess_example(self, example):
        """DOCSTRING."""  # TODO: Write docstring.
        inputs = self.tokenizer(
            example["sentence"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "token_type_ids": inputs["token_type_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": example["label"],
        }

    @torch.no_grad()
    def run_inference(self, token_type_ids, attention_mask):
        """DOCSTRING."""  # TODO: Write docstring.
        logits = self.model(
            token_type_ids, attention_mask=attention_mask
        ).last_hidden_state[0]
        return logits

    def extract_embeddings(self, dataset):
        """DOCSTRING."""  # TODO: Write docstring.
        results = {"outputs": [], "labels": []}
        for example in tqdm(dataset):
            example = self.preprocess_example(example)
            token_type_ids = example["token_type_ids"].to(self.device)
            attention_mask = example["attention_mask"].to(self.device)
            label = example["label"]
            outputs = self.run_inference(token_type_ids, attention_mask)
            results["outputs"].append(outputs.cpu())
            results["labels"].append(label)
        return results






