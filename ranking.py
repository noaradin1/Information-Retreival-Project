from collections import defaultdict, Counter
import numpy as np
from inverted_index_gcp import *
import math

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def read_posting_list(inverted, w,folder_name):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, folder_name)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def cosine_similarity_help(tokens, index, tf, idf):
    '''
    This function calculating the cosine similarity of the given query (tokens) and the documents of the index
    Parameters:
    -----------
    tokens: list, represting the tokens of the query.
    index: InvertedIndex, the index object (title/anchor/body)
    tf: dictionary, the term frequency dictionary for the tokens list
    idf: dictionary, the idf score of each token
    Returns:
    -----------
    sorted list of the relevant documents based on their cosine similarity ranking (based on the given query)
    '''

    tfidf = defaultdict(float)
    for token, idf_val in idf.items():
        for doc_id, tf_val in tf[token]:
            # adding tfidf of the token
            tfidf[doc_id] += tf_val * idf_val  
    # calculating cosine similarity
    tfidf = {d_id: tfidf[d_id] / (index.DL[d_id] * len(tokens)) for d_id in tfidf}
    return sorted(tfidf.items(), key= lambda x:x[1], reverse=True)
   

def tfidf(tokens, index, folder_name):
    '''
    This function calculating the tf-idf ranking of the given query (tokens)
    Parameters:
    -----------
    tokens: list, represting the tokens of the query.
    index: InvertedIndex, the index object (title/anchor/body)
    folder_name: string, represting the folder name of the index (used for reading bin files).
    Returns:
    -----------
    tf: the term frequency dictionary for the tokens list 
    idf: the idf score of each token
    '''

    tf, idf = defaultdict(float), defaultdict(float)
    n = len(index.DL)
    for token in tokens:
        try:
            # extract posting list of the token
            posting_list = read_posting_list(index, token, folder_name)
            
            # TF (normalized by document length)
            tf_values = {doc_id: tf_val / index.DL[doc_id] for doc_id, tf_val in posting_list}
            tf[token] = list(tf_values.items())
            
            # IDF
            df = len(posting_list)
            # calculating idf
            idf[token] = np.log2((n + 1) / (df + 1)) 

        except:
            continue

    return tf, idf


def cosine_similarity(query, index, folder_name):
    '''
    This function calculating the cosine similarity of the given query (tokens)
    Parameters:
    -----------
    tokens: list, represting the tokens of the query.
    index: InvertedIndex, the index object (title/anchor/body)
    folder_name: string, represting the folder name of the index (used for reading bin files).
    Returns:
    -----------
    sorted list of the relevant documents based on their cosine similarity ranking (based on the given query)
    '''

    tf, idf = tfidf(query, index, folder_name)
    return cosine_similarity_help(query, index, tf, idf)


def binary_ranking(tokens, index, folder_name):
    '''
    This function calculating the binary ranking of the given query (tokens)
    Parameters:
    -----------
    tokens: list, represting the tokens of the query.
    index: InvertedIndex, the index object (title/anchor/body)
    folder_name: string, represting the folder name of the index (used for reading bin files).
    Returns:
    -----------
    sorted list of the relevant documents based on their binary ranking (based on the given query)
    '''
    docs_terms = {}
    for token in tokens:
        # for every term that exists in the index
        if token in index.df.keys():
            # for every doc in posting list, add 1 to the docs dictionary
            for doc_id, tf in read_posting_list(index, token, folder_name):
                docs_terms[doc_id] = docs_terms.get(doc_id,0)+ 1
    # sort relevant documents based on number of terms
    return sorted(docs_terms.items(), key=lambda x: x[1], reverse=True)

    
 