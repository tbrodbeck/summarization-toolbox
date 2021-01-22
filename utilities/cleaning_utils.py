"""
this module provides functions
to clean text and summaries
"""


def clean_text(text: str) -> str:
    """basic text cleaning of spaces and newlines

    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    """
    character_list = list()
    text = text.replace('  ', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()
    for character in text:
        if character == ' ' \
                or character.isalpha() \
                or character.isdigit():
            character_list.append(character)

    return ''.join(character_list)


def truncate_incomplete_sentences(text: str, nlp) -> str:
    """remove incomplete sentences at the end of a summary

    Args:
        text (str): summary text
        nlp (spacy object): language model

    Returns:
        str: truncated summary
    """
    excluded_puncts = [",", ";"]
    doc = nlp(text)
    tokens = [token for token in doc]
    if tokens:
        last_token = tokens[-1]
        # check if last token is actually
        # an end of a sentence
        if last_token.is_punct and \
                last_token.text not in excluded_puncts:
            return text
        else:
            # remove the last, incomplete sentence
            sentences = [sent for sent in doc.sents]
            if len(sentences) < 2:
                return None
            else:
                return text[:sentences[-2].end_char]


def limit_data(data_dict: dict, limit: int = -1) -> dict:
    """apply limitation to data set

    Args:
        data_dict (dict): set with source and target
        limit (int, optional): limit of data to process.
        Defaults to -1.

    Returns:
        [dict]: dict with limited source and target
    """
    if limit == -1:
        return data_dict

    new_dict = dict()
    for item in ["source", "target"]:
        new_dict[item] = {
            "input_ids": data_dict[item]['input_ids'][:limit],
            "attention_mask": data_dict[item]['attention_mask'][:limit]
        }
    return new_dict
