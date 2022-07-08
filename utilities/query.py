# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import fasttext
import xml.etree.ElementTree as ET
import sys

# setting path
sys.path.append('../week3')

from normalize import normalize_query

# load categories
categories_file_name = '/Users/vitalii.mishchenko/Documents/personal/opensearch/data/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
tree = ET.parse(categories_file_name)
root = tree.getroot()
categories = []
names = []
for child in root:
    id = child.find('id').text
    name = child.find('name').text
    categories.append(id)
    names.append(name)
category_to_name_df = pd.DataFrame(list(zip(categories, names)), columns =['category', 'name'])


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None, synonyms = False, category_filter = None, category_boost=None):
    name_field = "name.synonyms" if synonyms else "name"
    name_analyzer = "custom_synonym" if synonyms else "standard"
    if synonyms:
        print(f"Search with synonyms")

    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                        ],
                        "should": [  #
                            {
                                "match": {
                                    name_field: {
                                        "query": user_query,
                                        "analyzer": name_analyzer,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": ["name^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath", "name.synonyms"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        },
        "_source": [
            "name",
            "_score"
        ]
    }

    if category_filter is not None:
        query_obj["query"]["function_score"]["query"]["bool"]["must"].append(category_filter)

    if category_boost is not None:
        query_obj["query"]["function_score"]["functions"].append(category_boost)

    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj

FASTEXT_FOLDER="/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext/query/"
MODEL_FILE_NAME="query_classifier.bin"
model = fasttext.load_model(f'{FASTEXT_FOLDER}{MODEL_FILE_NAME}')

def get_max_boost_category_list(categories, probabilities, boost_probability_threshold):
    boost_category_list = []

    for idx, category in enumerate(categories):
        if probabilities[idx] > boost_probability_threshold:
            category_name = category.replace("__label__", "")
            boost_category_list.append(category_name)
    return boost_category_list


def get_boost_category_list_by_max(categories, probabilities, boost_probability_threshold):
    boost_category_list = []

    for idx, category in enumerate(categories):
        if probabilities[idx] > boost_probability_threshold:
            category_name = category.replace("__label__", "")
            boost_category_list.append(category_name)
    return boost_category_list

def get_boost_category_list_by_sum(categories, probabilities):
    boost_category_list = []
    probability_sum = 0
    for idx, category in enumerate(categories):
        if probability_sum >= 1:
            break

        probability_sum += probabilities[idx]
        category_name = category.replace("__label__", "")
        boost_category_list.append(category_name)

    return boost_category_list

def get_boost_category(categories, probabilities, boost_probability_threshold):
    category_boost=None

    if probabilities.max() > boost_probability_threshold:
        method = "max"
        boost_category_list = get_boost_category_list_by_max(categories, probabilities, boost_probability_threshold)
    else:
        method = "sum"
        boost_category_list = get_boost_category_list_by_sum(categories, probabilities)

    print(f'- Boost {len(boost_category_list)} categories by using "{method}" method')
    if len(boost_category_list) > 0:
        for cat in boost_category_list:
            print(f'-- {category_to_name_df[category_to_name_df["category"] == cat]["name"].values[0]}')

        category_boost = {
            "filter": {
                "terms": {
                    "categoryPathIds.keyword": boost_category_list
                }
            },
            "weight": 5
        }

    return category_boost

def get_filter_category(categories, probabilities, filter_probability_threshold):
    category_filter = None

    filter_category_list = []
    for idx, category in enumerate(categories):
        if probabilities[idx] > filter_probability_threshold:
            category_name = category.replace("__label__", "")
            filter_category_list.append(category_name)

    print(f'- Filter {len(filter_category_list)} categories')
    if len(filter_category_list) > 0:
        for cat in filter_category_list:
            print(f'-- {category_to_name_df[category_to_name_df["category"] == cat]["name"].values[0]}')

        # "category_filter" is too restrictive in case when category was predicted incorrectly
        category_filter = {
            "terms": {
                "categoryPathIds.keyword": filter_category_list
            }
        }
    return category_filter

hardcoded_synonyms = {}
hardcoded_synonyms['irobot'] = ["vacuum", "robot", "cleaner"]

def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc", synonyms=False):
    # check whether to apply prediction?
    # p:iphone -    with prediction
    # iphone -      without prediction
    apply_prediction = False
    if user_query.startswith("p:"):
        apply_prediction = True
        user_query = user_query.replace("p:", "")

    # apply hard-coded synonyms at query time
    for word in hardcoded_synonyms:
        if word in user_query.lower():
            for synonym in hardcoded_synonyms[word]:
                if synonym not in user_query.lower():
                    user_query += " " + synonym

    # ultimately will be passed to create_query()
    category_boost=None
    category_filter=None

    # predict categories based on query and build:
    #   - category_boost
    #   - category_filter
    if apply_prediction:
        candidate_count = 4
        filter_probability_threshold = 0.95
        boost_probability_threshold = 0.5
        categories, probabilities = model.predict(normalize_query(user_query), k=candidate_count)

        category_boost = get_boost_category(categories, probabilities, boost_probability_threshold)
        category_filter = get_filter_category(categories, probabilities, filter_probability_threshold)

    query_obj = create_query(user_query, click_prior_query=None, filters=None, sort=sort, sortDir=sortDir, source=["name", "shortDescription"],
                             synonyms=synonyms,
                             category_filter=category_filter, category_boost=category_boost)
    # print(query_obj)
    try:
        response = client.search(query_obj, index=index)
        if response:
            print_results(response)
    except:
        print('Error during request')

def print_results(response):
    print(f'Total hits: {response.get("hits").get("total").get("value")}')

    hits = response.get('hits').get('hits')
    for hit in hits:
        print(f'-- {hit["_source"]["name"][0]} ({hit["_score"]})')

if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')

    general.add_argument('--synonyms', default=False, help='Include synonyms in search. If this is set, search will be performed on field indexed with synonyms')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    for line in fileinput.input(('-',)):
        query = line.rstrip()
        if query == "Exit":
            break
        search(client=opensearch, user_query=query, index=index_name, synonyms=args.synonyms)