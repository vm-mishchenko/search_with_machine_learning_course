import re
from nltk.stem import WordNetLemmatizer

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()
lemmatizer = WordNetLemmatizer()

normalize_cache = {}
normalize_count = {}
normalize_count['count'] = 0
def normalize_query(query):
  normalize_count['count'] += 1
  if normalize_count['count'] % 200000 == 0:
    print(f'Processed {normalize_count["count"]} queries.')

  if normalize_cache.get(query) is None:
    # remove special characters
    normalized_query = re.sub(r"[^a-zA-Z0-9 ]", "", query)

    # remove excessive spaces
    normalized_query = " ".join(normalized_query.split())

    # lowercase
    normalized_query = normalized_query.lower()

    # lemmatize
    tokens = nltk.word_tokenize(normalized_query)
    lema_token = []
    for token in tokens:
      if len(token) > 1:
        lema_token.append(lemmatizer.lemmatize(token))

    normalized_query = ' '.join(lema_token)

    # stemming (looks like lemmatizer works better)
    # normalized_query = stemmer.stem(normalized_query, to_lowercase=True)

    normalize_cache[query] = normalized_query

  return normalize_cache.get(query)