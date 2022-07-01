import nltk

# Part-of-speech tags
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

str = "Cats eat raw fish."
tokens = nltk.word_tokenize(str)
part_of_speach = nltk.pos_tag(tokens)

# Named entity recognition (NER)
nltk.download('words')
nltk.download('maxent_ne_chunker')
str = "Barack Obama served as the 44th President of the United States."
tokens = nltk.word_tokenize(str)
part_of_speach = nltk.pos_tag(tokens)
nltk.ne_chunk(part_of_speach)
