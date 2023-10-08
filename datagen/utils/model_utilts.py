"""DOCSTRING."""  # TODO: Write docstring.

from transformers import BertTokenizer, BertModel


def load_bert_model(model_name="bert-base-uncased"):
    """Load the BERT model from the HuggingFace Transformers library.

    Args:
        model_name (str, optional): _description_. Defaults to "bert-base-uncased".

    Returns:
        model: The BERT model.
    """
    bert_model = BertModel.from_pretrained(model_name)
    return bert_model


def load_bert_tokenizer(model_name="bert-base-uncased"):
    """Load the BERT tokenizer from the HuggingFace Transformers library.

    Args:
        model_name (str, optional): _description_. Defaults to "bert-base-uncased".

    Returns:
        tokenizer: The BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer


# TODO Add Other Models Here
