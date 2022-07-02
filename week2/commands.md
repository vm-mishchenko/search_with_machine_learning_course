```shell
# generate fasttext compatible file by running createContentTrainingData.py

FASTEXT_FOLDER="/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext"

# Shuffle product labels
shuf $FASTEXT_FOLDER/labeled_products.txt > $FASTEXT_FOLDER/shuffled_labeled_products.txt

# Split on training and test data
head -100000 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.train
tail -15504 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.test

# train model
# https://fasttext.cc/docs/en/supervised-tutorial.html
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier

# get prediction for your product name
fasttext predict $FASTEXT_FOLDER/product_classifier.bin -

# test model with test data
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result
# P@1, R@1 explanation: https://fasttext.cc/docs/en/supervised-tutorial.html#advanced-readers-precision-and-recall
N       15488 - evaluated using 15488 examples
P@1     0.585 - % of the time when our top-predicted label is a correct
R@1     0.585 - % of correctly predicted labels over number of actual labels

# test model: obtain 2 labels for P@5 and R@5
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test 5
P@5     0.159
R@5     0.793

# Increase number of epochs to 25
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier -epoch 25
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result
P@1     0.772
R@1     0.772

# Increase the learning rate to 1.0 and go back to 25 epochs
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier -epoch 25 -lr 1.0
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result
P@1     0.772
R@1     0.772

# Set word ngrams to 2 to learn from bigrams
# https://fasttext.cc/docs/en/supervised-tutorial.html#word-n-grams
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier -lr 1.0 -epoch 25 -wordNgrams 2
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result
N       15488
P@1     0.798
R@1     0.798

# Preprocess the text and recreate the training and test data
cat $FASTEXT_FOLDER/labeled_products.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > $FASTEXT_FOLDER/labeled_products.preprocessed.txt
shuf $FASTEXT_FOLDER/labeled_products.preprocessed.txt > $FASTEXT_FOLDER/shuffled_labeled_products.txt
head -100000 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.train
tail -15504 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.test
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier -lr 1.0 -epoch 25 -wordNgrams 2
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result
P@1     0.794
R@1     0.794

# re-generate initial product with stemming
# run createContentTrainingData.py, I've already added stemming code
shuf $FASTEXT_FOLDER/labeled_products.txt > $FASTEXT_FOLDER/shuffled_labeled_products.txt
head -100000 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.train
tail -15504 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.test
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier -lr 1.0 -epoch 25 -wordNgrams 2
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result
P@1     0.793
R@1     0.793

# re-generate product with stemming + keep only categories that has 500+ names
# looks into createContentTrainingData.py it has "min_products" param
shuf $FASTEXT_FOLDER/labeled_products.txt > $FASTEXT_FOLDER/shuffled_labeled_products.txt
head -100000 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.train
tail -15504 $FASTEXT_FOLDER/shuffled_labeled_products.txt > $FASTEXT_FOLDER/labeled_products.test
fasttext supervised -input $FASTEXT_FOLDER/labeled_products.train -output $FASTEXT_FOLDER/product_classifier -lr 1.0 -epoch 25 -wordNgrams 2
fasttext test $FASTEXT_FOLDER/product_classifier.bin $FASTEXT_FOLDER/labeled_products.test
# result is awesome, but categories without a lot of names are skipped
P@1     0.998
R@1     0.998

# TODO: The next level is to roll up infrequently used labels to their parent or other ancestor categories.
```

