import string as st
from nltk.corpus import stopwords
import nltk

nltk.download("punkt")
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def is_punctuation(c: str, punctuation=st.punctuation) -> bool:
    """
    Checks if a character is a punctuation character.
    """
    return c in punctuation


def is_stopword(w: str, stopword=stopwords.words("english")) -> bool:
    """
    Checks if a word is a stopword.
    """
    return w in stopword


def is_mention(w: str) -> bool:
    """
    Checks if a word is a mention.
    """
    return w[0] == "@"


def is_hashtag(w: str) -> bool:
    """
    Checks if a word is a hashtag.
    """
    return w[0] == "#"


def is_link(w: str) -> bool:
    """
    Checks if a word is a link.
    """
    return (w[0:4] == "www.") or (w[0:4] == "http")


def text_process(s: str) -> str:
    """
    Takes in a string of text, then performs the following:
    0. To lower case.
    1. Remove all punctuation.
    2. Remove all stopwords.
    3. Returns a string with the cleaned text.
    """
    wo_punc = "".join([c for c in s.lower() if not is_punctuation(c)])
    processed = " ".join([w for w in wo_punc.split() if not is_stopword(w)])
    return processed


def tokenize(s: str) -> str:
    """
    Tokenizes words in a given string.
    """
    tokenized_list = word_tokenize(s)
    return str(" ".join(tokenized_list))


def stem(s: str, ps=PorterStemmer()) -> list:
    """
    Stems given string of words, saves as a list.
    """
    words = s.split()
    return [ps.stem(w) for w in words]
