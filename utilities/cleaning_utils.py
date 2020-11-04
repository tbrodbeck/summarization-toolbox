"""
this module provides functions
to clean text and summaries
"""

def clean_text(text: str) -> str:
    """
    remove non alphas from text
    :param text:
    :return:
    """
    character_list = list()
    text = text.replace('  ', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()
    for t in text:
        if t == ' ' or t.isalpha() or t.isdigit():
            character_list.append(t)

    return ''.join(character_list)


def truncate_incomplete_sentences(text: str, nlp) -> str:
    """
    make sure no incomplete
    sentences are at the end
    of a summary
    :param nlp:
    :param text:
    :return:
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
                return text
            else:
                return text[:sentences[-2].end_char]
