import os
import argparse
from pathlib import Path
from nltk.stem import SnowballStemmer
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

stemmer = SnowballStemmer("english")

def transform_training_data(title, comment):
    # IMPLEMENTED
    tokens = nltk.word_tokenize(comment)
    # https://pythonexamples.org/nltk-pos-tagging/
    part_of_speach = nltk.pos_tag(tokens)
    # JJ* Adjective
    # NN* Noun
    # RB* Adverb
    # VB* Adverb
    filtered_tokens = []
    for (word, part) in part_of_speach:
        if part.startswith("JJ") or part.startswith("NN") or part.startswith("RB") or part.startswith("VB"):
            filtered_tokens.append(word)

    stemmed_comment = stemmer.stem(' '.join(filtered_tokens))

    return stemmed_comment


# Directory for review data
# directory = r'/workspace/datasets/product_data/reviews/'
directory = r'/Users/vitalii.mishchenko/Documents/personal/opensearch/data/product_data/reviews/'
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing reviews")
general.add_argument("--output", default=r'/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext/reviews.txt',
                     help="the file to output to")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input


print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            with open(os.path.join(directory, filename)) as xml_file:
                for line in xml_file:
                    if '<rating>'in line:
                        rating = float(line[12:15])
                        grade = 'unknown'

                        if rating <= 2:
                            grade = 'negative'
                        elif rating > 2 and rating <= 4:
                            grade = 'neutral'
                        else:
                            grade = 'positive'

                    elif '<title>' in line:
                        title = line[11:len(line) - 9]
                    elif '<comment>' in line:
                        comment = line[13:len(line) - 11]
                    elif '</review>'in line:
                      output.write("__label__%s %s\n" % (grade, transform_training_data(title, comment)))
