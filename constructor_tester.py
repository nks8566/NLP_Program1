
from CorpusReader_SLM import CorpusReader_SLM
import nltk
from nltk.corpus import brown

# Make sure the required corpora are downloaded
nltk.download('brown')
nltk.download('stopwords')

# Use the Brown corpus as a test
corpus = brown

# Create your CorpusReader_SLM object
reader = CorpusReader_SLM(corpus, stopWord="standard", toStem=False, smooth=True, trigram=True)

# Print some basic checks
print("Most common unigrams:")
for word, count in reader.unigram_counts.most_common(20):
    print(word, count, round(reader.unigram_probs[word], 4))

print("\nMost common bigrams:")
for bigram, count in list(reader.bigram_counts.most_common(0)):
    bigram_str = " ".join(bigram)
    print(bigram_str, count, round(reader.bigram_probs[bigram_str], 4))

if reader.trigram_flag:
    print("\nMost common trigrams:")
    for trigram, count in list(reader.trigram_counts.most_common(0)):
        trigram_str = " ".join(trigram)
        print(trigram_str, count, round(reader.trigram_probs[trigram_str], 6))

# Test the unigram function (returns x number of unigrams by probability)
print("\nTop 5 unigrams by probability:")
for word, prob in reader.unigram(5):
    print(word, round(prob, 4))

print("\nTop 10 unigrams by probability:")
for word, prob in reader.unigram(10):
    print(word, round(prob, 4))

print("\nTop 20 unigrams by probability:")
for word, prob in reader.unigram(20):
    print(word, round(prob, 4))
