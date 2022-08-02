from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import nltk
import string


def process_tokens(toks):
    # fill in the functions: process_tokens_1,
    # process_tokens_2, and process_tokens_3 functions and
    # uncomment the one you want to test below

    # NOTE: make sure to switch back to process_tokens_original
    # and rebuild the index before
    # tackling the other assignment questions

    # return process_tokens_1(toks)
    return process_tokens_2(toks)
    # return process_tokens_3(toks)
    return process_tokens_original(toks)


# get the nltk stopwords list
stopwords = set(nltk.corpus.stopwords.words("english"))


def process_tokens_original(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        new_toks.append(t)
    return new_toks


ps = PorterStemmer()


def process_tokens_1(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue

        # Use PorterStemmer for stemming
        new_toks.append(ps.stem(t))
    return new_toks


wl = WordNetLemmatizer()


def word_pos(pos):
    if pos.startswith("J"):
        return "a"
    elif pos.startswith("V"):
        return "v"
    elif pos.startswith("R"):
        return "r"
    else:
        return "n"


def process_tokens_2(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    toks_w_tags = pos_tag(toks)
    for t, pos in toks_w_tags:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue

        new_toks.append(wl.lemmatize(t.lower(), pos=word_pos(pos)))
    return new_toks


def process_tokens_3(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue

        if t in string.punctuation:
            continue
        new_toks.append(t)
    return new_toks


def tokenize_text(data):
    """Convert a document as a string into a document as a list of
    tokens. The tokens are strings.

    Args:
        data (str): The input document

    Returns:
        list(str): The list of tokens.
    """
    # split text on spaces
    tokens = data.split()
    return tokens
