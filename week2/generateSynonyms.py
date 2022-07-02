import fasttext
import pandas as pd
import csv

FASTEXT_FOLDER="/Users/vitalii.mishchenko/Documents/personal/opensearch/data/fasttext"
MODEL_FILE_NAME = '/title_model.bin'
TOP_WORDS_FILE_NAME = '/top_words.txt'
OUTPUT_SYNONYMS_FILE_NAME = '/synonyms.csv'

model = fasttext.load_model(f'{FASTEXT_FOLDER}{MODEL_FILE_NAME}')
threshold = 0.75
df = pd.read_csv(f'{FASTEXT_FOLDER}{TOP_WORDS_FILE_NAME}', sep='\t', header=None, names=["token"])
tokens = df['token'].tolist()
synonyms_dict = []

for token in tokens:
  synonyms_data = model.get_nearest_neighbors(token)
  nn = [token]
  for (similarity, synonym) in synonyms_data:
    if similarity > threshold:
      nn.append(synonym)
  if len(nn) > 1:
    synonyms_dict.append({"synonym": ', '.join(nn)})
synonym_df = pd.DataFrame(synonyms_dict)
synonym_df.to_csv(f'{FASTEXT_FOLDER}{OUTPUT_SYNONYMS_FILE_NAME}', header=False, index=False,
                  sep='\t', quoting = csv.QUOTE_NONE)