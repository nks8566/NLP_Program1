
import random
import numpy as np

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
            if trigram:
                processed_sent = ['<s>', '<s>'] + processed_sent + ['</s>']
            else:
                processed_sent = ['<s>'] + processed_sent + ['</s>']
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
                    self.trigram_probs[trigram_str] = (count + 1) / (self.bigram_counts[(trigram[0], trigram[1])] + V)
        else:
            total_unigrams = sum(self.unigram_counts.values())
            # probabilities without smoothing
            self.unigram_probs = {word: count / total_unigrams for word, count in self.unigram_counts.items()}
            self.bigram_probs = {" ".join(bigram): count / self.unigram_counts[bigram[0]] for bigram, count in self.bigram_counts.items()}
            if trigram:
                self.trigram_probs = {" ".join(trigram): count / self.bigram_counts[(trigram[0], trigram[1])]
                for trigram, count in self.trigram_counts.items()}

    # Class methods
    def unigram(self, count = 0):
        # self.unigram_probs -- dictionary containing unigram probabilities, {key: value} as {word: probability}
        # self.unigram_counts -- counter object, ie a dictionary containing unigram frequency, {key: value} as {word: count}

        if count > 0:
            # sort by probability in descending order, then select by highest probability
            sorted_by_prob = sorted(self.unigram_probs.items(), key=lambda x: x[1], reverse=True)
            selected_items = sorted_by_prob[:count]
        else:
            # use all unigrams
            selected_items = list(self.unigram_probs.items())
        
        # sort alphabetically by word
        result = sorted(selected_items, key=lambda x: x[0])
        return result

    def bigram(self, count = 0):
        if count > 0:
            # sort by probability in descending order, then select by highest probability
            sorted_by_prob = sorted(self.bigram_probs.items(), key=lambda x: x[1], reverse=True)
            selected_items = sorted_by_prob[:count]
        else:
            # use all bigrams
            selected_items = list(self.bigram_probs.items())
        
        # sort alphabetically by word
        result = sorted(selected_items, key=lambda x: x[0])
        return result

    def trigram(self, count = 0):
        if count > 0:
            sorted_by_prob = sorted(self.trigram_probs.items(), key=lambda x: x[1], reverse=True)
            selected_items = sorted_by_prob[:count]
        else:
            selected_items = list(self.trigram_probs.items())

        # sort alphabetically by word
        result = sorted(selected_items, key=lambda x: x[0])
        return result

    def unigramGenerate(self, code = 0, head = []):

        # Return empty list if code is not 0, 1, or 2
        if code not in [0,1,2]:
            return []

        generatedSentence = list(head) if head else []

        # Iterate until ending punctuation is appended, or if generatedSentence is initially empty
        while not generatedSentence or generatedSentence[-1] not in [".", "!", "?"]:

            if code == 0:

                # Find the highest probability from the unigram distribution
                highestProbability = max(self.unigram_probs.values())

                # Create a list that stores all tiebreaker words (if there is a tiebreaker, else returns only element)
                tiebreakerValues = [key for key, value in self.unigram_probs.items() if value == highestProbability]

                # Ternary to return a random choice from tiebreaker words if more than 1 element exists, else return only element
                selectedWord = random.choice(tiebreakerValues) if len(tiebreakerValues) > 1 else tiebreakerValues[0]
                
            elif code == 1:

                # Stores non-zero probas in a new dictionairy
                nonZeroProbaDict = {word: probability for word, probability in self.unigram_probs.items() if probability != 0}

                words = list(nonZeroProbaDict.keys())
                probabilities = list(nonZeroProbaDict.values())

                # Normalize probabilities
                totalProba = sum(probabilities)
                probabilities = [p / totalProba for p in probabilities]

                # Selected a random word using weighted sampling (probability of selecting word w, is based on w's probability)
                # Assumes probas sum to 1 in the unigram_probs dict
                selectedWord = np.random.choice(words, p=probabilities)
                
            elif code == 2:

                # Stores list of tuples ('word', 'probability') in descending order
                sortedProbas = sorted(self.unigram_probs.items(), key=lambda x: x[1], reverse=True)

                
                if len(sortedProbas) <= 10:
                    probasToUse = sortedProbas
                else:
                    # Maps to 10th word's probability
                    lastProbaTiebreaker = sortedProbas[9][1]
                    probasToUse = [p for p in sortedProbas if p[1] >= lastProbaTiebreaker]

                # Store words, and probabilities
                words = [word for word, proba in probasToUse]
                probabilities = [proba for word, proba in probasToUse]

                # Find scalar to normalize probabilities
                probScale = 1 / sum(probabilities)

                # Normalize probas
                normalizedProbabilities = [probScale * p for p in probabilities]

                # Pick random element using weighted sampling
                selectedWord = np.random.choice(words, p=normalizedProbabilities)

            # Append selected token
            generatedSentence.append(selectedWord)

        
        formattedSentence = ""

        for idx, token in enumerate(generatedSentence):
            
            if token in [",", ".", "!", "?"]:
                formattedSentence += token
            else:
                if idx == 0:
                    formattedSentence += token
                else:
                    formattedSentence += f" {token}"


        # Missing formatting
        return formattedSentence

    def bigramGenerate(code = 0, head= []):
        pass

    def trigramGenerate(code = 0, head= []):
        pass
