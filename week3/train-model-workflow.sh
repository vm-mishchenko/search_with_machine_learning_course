echo "-----Generate labeled queries"
/Users/vitalii.mishchenko/Documents/experiments/search_with_machine_learning_course/venv/bin/python /Users/vitalii.mishchenko/Documents/experiments/search_with_machine_learning_course/week3/create_labeled_queries.py

FASTEXT_FOLDER="/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext/query"

echo "-----Shuffle data"
shuf $FASTEXT_FOLDER/labeled_query_data.txt > $FASTEXT_FOLDER/shuffled_labeled_query_data.txt

echo "-----Generate Train and Test data"
head -50000 $FASTEXT_FOLDER/shuffled_labeled_query_data.txt > $FASTEXT_FOLDER/labeled_query_data.train
tail -20000 $FASTEXT_FOLDER/shuffled_labeled_query_data.txt > $FASTEXT_FOLDER/labeled_query_data.test

echo "-----Train model"
fasttext supervised -input $FASTEXT_FOLDER/labeled_query_data.train -output $FASTEXT_FOLDER/query_classifier -lr 0.5 -epoch 25 -wordNgrams 2

echo "-----Test model"
fasttext test $FASTEXT_FOLDER/query_classifier.bin $FASTEXT_FOLDER/labeled_query_data.test