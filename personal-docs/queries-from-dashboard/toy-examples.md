```shell
# Is LTR store exist?
GET _ltr/searchml_ltr


# Feature set inside "searchml_ltr" ltr store
GET /_ltr/searchml_ltr/_featureset/ltr_toy


# Rescore results using trained model
GET /searchml_ltr/_search
{
  "query": {
    "multi_match": {
      "query": "friend",
      "fields": [
        "title^2",
        "body"
      ]
    }
  },
  "rescore": {
    "window_size": 10,
    "query": {
      "rescore_query": {
        "sltr": {
          "params": {
            "keywords": "dogs"
          },
          "model": "ltr_toy_model",
          "store": "searchml_ltr",
          "active_features": [
            "title_query",
            "body_query",
            "price_func"
          ]
        }
      },
      "rescore_query_weight": "2"
    }
  }
}
```