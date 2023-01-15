import re
from nltk.corpus import stopwords
import gensim.downloader as api
from collections import Counter

model = api.load("glove-wiki-gigaword-50")


english_stopwords = frozenset(stopwords.words('english'))
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
our = ["why","how","what","make"]
all_stopwords = english_stopwords.union(corpus_stopwords)
our_stopwords = all_stopwords.union(our)


# from hw1
def get_html_pattern():
  return "(\<[^\<]+\>)"

def get_date_pattern():
  m_31_days = 'January|Jan|March|Mar|May|July|Jul|August|Aug|October|Oct|December|Dec'
  d_31_days = '([1-9]|3[10]|[12]\d)'
  m_30_days = 'April|Apr|June|Jun|September|Sep|November|Nov'
  d_30_days = '([1-9]|30|[12]\d)'
  m_29_days = 'February|Feb'
  d_29_days = '([1-9]|[12]\d)'
  year = '\d*'
  return f"(?i)(({m_31_days})\s{d_31_days}\,\s{year})|({d_31_days}\s({m_31_days})\s{year})|(({m_30_days})\s{d_30_days}\,\s{year})|({d_30_days}\s({m_30_days})\s{year})|(({m_29_days})\s{d_29_days}\,\s{year})|({d_29_days}\s({m_29_days})\s{year})"
def get_time_pattern():
  secmin = '([0-5]\d)'
  hours = '([0-1]\d)'
  no_chars = '(?![A-Za-z0-9])'
  no_chars_start = '(?<![A-Za-z0-9])'
  return f"({hours}[\.]{secmin}(AM|PM){no_chars})|({hours}{secmin}(p\.m\.|a\.m\.){no_chars})|{no_chars_start}([1-9]\:{secmin}\:{secmin}){no_chars}|{no_chars_start}(([1]\d)\:{secmin}\:{secmin}){no_chars}|{no_chars_start}(([2][0-3])\:{secmin}\:{secmin}){no_chars}"

def get_percent_pattern():
    return "(?:(?<=\s|\())[\+\-]?\d{1,3}(,\d{3})*(\.\d+)?[%](?=(|\,|\.|\:)?(\s|$))"

def get_number_pattern():
    return "(?:(?<=\s|\())[\+\-]?\d{1,3}(,\d{3})*(\.\d+)?(?=(|\,|\.|\:)?(\s|$))|^\d+(?=\s)"

def get_word_pattern():
    return "(?:(?<=\s)|(?<=^))(([a-zA-Z]+[\']*[a-zA-Z]*)+\-*)*([a-zA-Z]+[\']*[a-zA-Z]*)?"

RE_TOKENIZE = re.compile(rf"""
(
    # parsing html tags
     (?P<HTMLTAG>{get_html_pattern()})                                  
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+) 
    # everything else
    |(?P<OTHER>.))""",  re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords based on our stop words list.
    Parameters:
    -----------
    text: string , represting the text to tokenize.
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [token for token in tokens if token not in our_stopwords]

def query_extension(query):
    """
    This function extend the given query using glove model (adding similar words to the query)
    Parameters:
    -----------
    text: string , represting the text to tokenize.
    Returns:
    -----------
    extended list of tokens (e.g., list of tokens).
    """

    extend_query = []
    for token in query:
        try:
            to_add = []
            similar = model.most_similar(token)
            if len(similar) >= 1:
                to_add.append(similar[0][0])
                if len(similar) > 1:
                    to_add.append(similar[1][0])
        except:
            continue
        extend_query.extend(to_add)
    return query + extend_query


def hw3_tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.
    Parameters:
    -----------
    text: string , represting the text to tokenize.
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens_no_stop = [token for token in tokens if token not in all_stopwords]
    return tokens_no_stop
