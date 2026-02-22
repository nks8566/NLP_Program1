import random
import numpy as np
from nltk.corpus import treebank
from CorpusReader_SLM import CorpusReader_SLM

# -----------------------------
# Initialize corpus with trigram support
# -----------------------------
myCorpus = CorpusReader_SLM(treebank, stopWord="none", toStem=False, smooth=False, trigram=True)

# -----------------------------
# Test n-gram probability methods
# -----------------------------
print("==== UNIGRAM TEST ====")
unigrams_all = myCorpus.unigram()
unigrams_top5 = myCorpus.unigram(5)
print(f"Total unigrams: {len(unigrams_all)}")
print("Top 5 unigrams (alphabetically):", unigrams_top5[:5])

print("\n==== BIGRAM TEST ====")
bigrams_all = myCorpus.bigram()
bigrams_top5 = myCorpus.bigram(5)
print(f"Total bigrams: {len(bigrams_all)}")
print("Top 5 bigrams (alphabetically):", bigrams_top5[:5])

print("\n==== TRIGRAM TEST ====")
trigrams_all = myCorpus.trigram()
trigrams_top5 = myCorpus.trigram(5)
print(f"Total trigrams: {len(trigrams_all)}")
print("Top 5 trigrams (alphabetically):", trigrams_top5[:5])

# -----------------------------
# Test sentence generation
# -----------------------------
print("\n==== UNIGRAM GENERATE ====")
for code in [0,1,2,99]:  # including invalid code
    sentence = myCorpus.unigramGenerate(code)
    print(f"Code {code}: {sentence}")

print("\n==== BIGRAM GENERATE ====")
for code in [0,1,2,99]:
    sentence = myCorpus.bigramGenerate(code)
    print(f"Code {code}: {sentence}")

print("\n==== TRIGRAM GENERATE ====")
for code in [0,1,2,99]:
    sentence = myCorpus.trigramGenerate(code)
    print(f"Code {code}: {sentence}")

# -----------------------------
# Optional: Test with head input
# -----------------------------
print("\n==== TEST HEAD INPUT ====")
head = ["the", "market"]
print("Unigram generate with head:", myCorpus.unigramGenerate(1, head))
print("Bigram generate with head:", myCorpus.bigramGenerate(1, head))
print("Trigram generate with head:", myCorpus.trigramGenerate(1, head))