
import nltk
from nltk.util import ngrams  
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter

class CorpusReader_SLM():
    def __init__(self, corpus, stopWord = "none", toStem = False, smooth = False, trigram = False):
        # corpus (required): a corpus object in NLTK
        self.toStem = toStem
        self.trigram_flag = trigram
        # stopWord: what stopWords to remove
        # “none”: no stopwords need to be removed
        if stopWord == "none":
            self.stopwords = set()
        # “standard”: use the standard English stopWord available in NLTK
        elif stopWord == "standard":
            self.stopwords = set(stopwords.words('english'))
        # Others: the input should be treated as a filename where stopwords are to be read. You
        # should assume any word inside the stopwords file is a stopword. You should assume the
        # file is being stored in the same directory as where the program is run.
        # Notice all stopwords (as in the corpus) should be converted to lower case (see below)
        else:
            with open(stopWord, 'r') as f:
                self.stopwords = set(line.strip().lower() for line in f)
        # toStem: if true, use the Snowball stemmer to stem the words beforehand.
        if toStem:
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = None
        # initialize counters
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter() # only used if trigram=True
        for sent in corpus.sents():
            processed_sent = []
            for word in sent:
                word = word.lower()
                if word not in self.stopwords:
                    if self.stemmer:
                        word = self.stemmer.stem(word)
                    processed_sent.append(word)
            # Update ngram counts
            self.unigram_counts.update(processed_sent)
            self.bigram_counts.update(ngrams(processed_sent, 2))
            if trigram:
                self.trigram_counts.update(ngrams(processed_sent, 3))
        # smooth: if true, use Laplace (+1) smoothing to calculate the probabilities.
        if smooth:
            self.vocab = set(self.unigram_counts.keys())
            V = len(self.vocab)
            N = sum(self.unigram_counts.values())
            # probabilities with Laplace
            self.unigram_probs = {}
            for word in self.vocab:
                self.unigram_probs[word] = (self.unigram_counts[word] + 1) / (N + V)
            self.bigram_probs = {}
            for bigram, count in self.bigram_counts.items():
                bigram_str = " ".join(bigram)
                self.bigram_probs[bigram_str] = (count + 1) / (self.unigram_counts[bigram[0]] + V)
            # trigram: if true, calculate and store trigram probabilities in addition to unigram and bigrams.
            if trigram:
                self.trigram_probs = {}
                for trigram, count in self.trigram_counts.items():
                    trigram_str = " ".join(trigram)
                    bigram_str = " ".join(trigram[:2])
                    self.trigram_probs[trigram_str] = (count + 1) / (self.bigram_counts[bigram_str] + V)
        else:
            # probabilities without smoothing
            self.unigram_probs = {word: count / sum(self.unigram_counts.values()) for word, count in self.unigram_counts.items()}
            self.bigram_probs = {" ".join(bigram): count / self.unigram_counts[bigram[0]] for bigram, count in self.bigram_counts.items()}
            if trigram:
                self.trigram_probs = {" ".join(trigram): count / self.bigram_counts[" ".join(trigram[:2])] for trigram, count in self.trigram_counts.items()}

    # Class methods
    def unigram(count = 0):
        pass

    def bigram(count = 0):
        pass

    def trigram(count = 0):
        pass

    def unigramGenerate(code = 0, head= []):
        pass

    def bigramGenerate(code = 0, head= []):
        pass

    def trigramGenerate(code = 0, head= []):
        pass
