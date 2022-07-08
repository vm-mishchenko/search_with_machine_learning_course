import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import re
import numpy as np
import csv
from nltk.stem import WordNetLemmatizer
from normalize import normalize_query

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()
lemmatizer = WordNetLemmatizer()

# categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
categories_file_name = '/Users/vitalii.mishchenko/Documents/personal/opensearch/data/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

# queries_file_name = r'/workspace/datasets/train.csv'
queries_file_name = r'/Users/vitalii.mishchenko/Documents/personal/opensearch/data/train.csv'
# output_file_name = r'/workspace/datasets/labeled_query_data.txt'
output_file_name = r'/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext/query/labeled_query_data.txt'
normalized_query_data_file_name = r'/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext/query/normalized_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=15000,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries = pd.read_csv(queries_file_name)[['category', 'query']]
queries = queries[queries['category'].isin(categories)]
queries_category_gb = queries.groupby('category')

# IMPLEMENTED: Convert queries to lowercase, and optionally implement other normalization, like stemming.
print('Normalize queries')
queries['query_normalized'] = queries['query'].map(normalize_query)
queries = queries[queries['query_normalized'].map(len) > 0]
queries.to_csv(normalized_query_data_file_name, header=False, sep=',', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

print('Read Normalized Query data...')
queries = pd.read_csv(normalized_query_data_file_name, sep=',', header=None, names=["category", "query", "query_normalized"])

# IMPLEMENTED: Roll up categories to ancestors to satisfy the minimum number of queries per category.
category_clicks_cache = {}
def get_category_clicks(category):
    if category_clicks_cache.get(category) is None:
        total_clicks = 0

        # in recursive case, child category might not be clicked
        if queries_category_gb.groups.get(category) is not None:
            total_clicks += len(queries_category_gb.get_group(category).index)

        # add all child categories clicks
        child_categories = parents_df[parents_df['parent'] == category]

        for idx, row in child_categories.iterrows():
            total_clicks += get_category_clicks(row['category'])

        category_clicks_cache[category] = total_clicks

    return category_clicks_cache.get(category)

parents_lookup = dict(zip(parents_df.category, parents_df.parent))
parents_lookup[root_category_id] = root_category_id

category_index_cache = {}
category_clicks_count2 = {}
category_clicks_count2['count'] = 0
def rollup_category2(category, previous_query_count=0, first_call=True):
    # first call
    if first_call is True:
        # track progress
        category_clicks_count2['count'] += 1
        if category_clicks_count2['count'] % 200000 == 0:
            print(f'Processed {category_clicks_count2["count"]} queries.')

    # calculate value
    total = previous_query_count

    # get current category queries count
    if category_index_cache.get(category) is None:
        # when it's recursive case, parent category might not have clicks
        if queries_category_gb.groups.get(category) is not None:
            category_index_cache[category] = len(queries_category_gb.get_group(category).index)
        else:
            category_index_cache[category] = 0

    total += category_index_cache.get(category)

    if total < min_queries:
        if category == root_category_id:
            #print(f'Warning: {original_category} category from required {min_queries} queries has only {total}')
            return root_category_id
        else:
            # recursive case
            return rollup_category2(parents_lookup[category], total, False)
    else:
        # base case
        return category

print('Roll up categories...')
print(f'Total {len(queries["category"].index)} queries')

# Convert such input with 100 as min value:
#      cat1(90 queries)
#  cat2(1)        cat3(200)
#             cat4(1)    cat4(1)
#
#
# to this output:
#     cat1(91)
#           cat3(202)
queries['category'] = queries['category'].apply(rollup_category2)

def prune_categories(queries, min_queries):
    grouped = queries.groupby('category').count().reset_index()[['category', 'query']]
    grouped.columns = ['category', 'num']
    categories_with_not_enough_queries = set(grouped[grouped.num < min_queries].category.values)

    if len(categories_with_not_enough_queries) > 0:
        print(f'Prune {len(categories_with_not_enough_queries)} categories that have less than {min_queries} queries:')
        for cat in categories_with_not_enough_queries:
            print(f'-- {cat}')

    return queries[~queries['category'].isin(categories_with_not_enough_queries)]

# there is still might be the category with less than "min_queries" queries
# delete them
queries = prune_categories(queries, min_queries)
print(f'{len(queries.groupby("category"))} unique categories')

def assertNoCategoriesToRollUpLeft(queries, min_queries):
    print('Assert valid roll up process.')
    if min_queries is None:
        raise Exception(f'min_queries cannot be None')

    grouped = queries.groupby('category').count().reset_index()[['category', 'query']]
    grouped.columns = ['category', 'num']
    categories_to_roll = set(grouped[grouped.num < min_queries].category.values)

    if len(categories_to_roll) != 0:
        print(f'{len(categories_to_roll)} categories have less than required number of queries')
        for category in categories_to_roll:
            print(f'{category} has {category_clicks_cache[category]}')

        raise Exception(f'{len(categories_to_roll)} have less than required number of queries')

assertNoCategoriesToRollUpLeft(queries, min_queries)

print('Create labels in fastText format')
queries['label'] = '__label__' + queries['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries = queries[queries['category'].isin(categories)]
queries['output'] = queries['label'] + ' ' + queries['query_normalized']
queries[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
