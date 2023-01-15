# Information-Retreival-Project

This repository contains the code for a search engine built for an Information Retrieval course. The search engine allows users to search for documents within the English Wikipedia corpus. 

### Files Included
1. `search_frontend.py` - contains the 6 methods required to implement the search functionality: search, search according to document title, search according to document body, search according to document anchor text, get page views and get page rank functions. 
2. `query_preprocess.py` - contains the query extension, query tokenization methods for the results retrieval.
3. `ranking.py` - contains the methods used for ranking the documents: binary ranking, tf-idf, cosine similarity
4. `inverted_index_gcp.py` - contains the inverted index file and all its methods.
5. `bucket_files.txt` - contains the files list from the buckets as required.

### How to use
The search engine can be used by running the `search_frontend.py` file.
