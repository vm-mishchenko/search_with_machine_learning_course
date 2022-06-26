import query_utils as qu
import ltr_utils as lu
import pandas as pd
import argparse
from opensearchpy import OpenSearch

# query_type='simple_query_obj'
# query_type='hand_tuned_query_obj'
# query_type='ltr_simple_query_obj'
query_type='ltr_hand_query_obj'

query = 'iphone4'
main_query_weight=0
rescore_query_weight=2


def evaluate_test_set(test_data, prior_clicks_df, opensearch, main_query_weight, rescore_query_weight, query_type):
  query_test_gb = test_data.groupby("query", sort=False) #small
  prior_clicks_gb = prior_clicks_df.groupby(["query"]) #large
  source = ["sku", "name"]
  rescore_size=500
  size=10
  xgb_model_name = 'ltr_model'
  ltr_store = 'week1'
  index = 'bbuy_products'

  for query_test in query_test_gb.groups.keys():
    prior_doc_ids = None
    prior_doc_id_weights = None
    query_times_seen = 0 # careful here
    most_clicked_category_id = None
    try:
      # select all clicks for particular query from `train` dataset
      prior_clicks_for_query = prior_clicks_gb.get_group(query_test)
      if prior_clicks_for_query is not None and len(prior_clicks_for_query) > 0:
        prior_doc_ids = prior_clicks_for_query.sku.drop_duplicates()
        # histogram gives us the click counts for all the doc_ids
        prior_doc_id_weights = prior_clicks_for_query.sku.value_counts()
        query_times_seen = prior_clicks_for_query.sku.count()

        most_clicked_category_id = qu.create_most_clicked_category_id(prior_clicks_for_query)
    except KeyError as ke:
      # nothing to do here, we just haven't seen this query before in our training set
      pass

    click_prior_query = qu.create_prior_queries(prior_doc_ids, prior_doc_id_weights, query_times_seen)

    # Run simple
    simple_query_obj = qu.create_simple_baseline(query_test, click_prior_query, filters=None, size=size, highlight=False, include_aggs=False, source=source)


    # Run hand-tuned
    hand_tuned_query_obj = qu.create_query(query_test, click_prior_query, filters=None, size=size, highlight=False, include_aggs=False, source=source)


    # Run LTR simple
    # NOTE: very important, we cannot look at the test set for click weights, but we can look at the train set.
    ltr_simple_query_obj = lu.create_rescore_ltr_query(query_test, simple_query_obj, click_prior_query, most_clicked_category_id, xgb_model_name, ltr_store, rescore_size=rescore_size,
                                                       main_query_weight=main_query_weight, rescore_query_weight=rescore_query_weight)


    # Run LTR hand-tuned
    ltr_hand_query_obj = lu.create_rescore_ltr_query(query_test, hand_tuned_query_obj, click_prior_query, most_clicked_category_id, xgb_model_name, ltr_store,
                                                     rescore_size=rescore_size, main_query_weight=main_query_weight, rescore_query_weight=rescore_query_weight)

    query_obj = None
    if query_type == 'simple_query_obj':
      query_obj = simple_query_obj

    if query_type == 'hand_tuned_query_obj':
      query_obj = hand_tuned_query_obj

    if query_type == 'ltr_simple_query_obj':
      query_obj = ltr_simple_query_obj

    if query_type == 'ltr_hand_query_obj':
      query_obj = ltr_hand_query_obj

    response = opensearch.search(body=query_obj, index=index)
    print_results(response)


def print_results(response):
  hits = response.get('hits').get('hits')

  for hit in hits:
    print(hit['_source']['name'][0])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run search queries.')
  general = parser.add_argument_group("general")
  general.add_argument('--train_file', default="train.csv",
                         help='Where to load the training file from under the output_dir.  Required when using --create_xgb_training')
  args = parser.parse_args()

  opensearch = OpenSearch(
      hosts=[{'host': 'localhost', 'port': 9200}],
      http_compress=True,  # enables gzip compression for request bodies
      http_auth=('admin', 'admin'),
      use_ssl=True,
      verify_certs=False,  # set to true if you have certs
      ssl_assert_hostname=False,
      ssl_show_warn=False,
  )

  train_df = pd.read_csv(args.train_file, parse_dates=['click_time', 'query_time'])
  test_data = pd.DataFrame(data=[query],
                    columns=['query'],
                    index=[0])

  evaluate_test_set(test_data, train_df, opensearch, main_query_weight, rescore_query_weight, query_type)