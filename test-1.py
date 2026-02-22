from nltk.corpus import PlaintextCorpusReader, treebank
from CorpusReader_SLM import *

#import nltk
#nltk.download('treebank')


'''
print(len(inaugural.words()))
print(inaugural.sents())
print(len(inaugural.sents()))
print(inaugural.fileids())
print(inaugural.sents(['1789-washington.txt']))

print(len(brown.words()))
print(brown.sents())
'''
print(len(treebank.words()))
print(treebank.sents())

'''
myCorpus = CorpusReader_SLM(inaugural)

print(myCorpus.unigram())
print(myCorpus.bigram())
print(myCorpus.unigramGenerate())
print(myCorpus.unigramGenerate(10, [This]))
'''


#  This is for testing your own corpus
#
#  create a set of text files, store them in a directory specified from 'rootDir' variable
#
#  

# rootDir = '_temp1'   # change that to the directory where the files are

# newCorpus = PlaintextCorpusReader(rootDir, '.*txt')
# x = newCorpus.sents()
# for y in x:
#     print(y)

# '''
# myC2 = CorpusReader_TFIDF(newCorpus)


# print("-----\n")

# '''

from nltk.corpus import treebank

myCorpus = CorpusReader_SLM(treebank, trigram=True)

print("########## UNIGRAM ##########")
print(myCorpus.unigram(10))

print("########## BIGRAM ##########")
print(myCorpus.bigram(10))

print("########## UNIGRAM GENERATE (code 0) ##########")
print(myCorpus.unigramGenerate(0))

print("########## UNIGRAM GENERATE (code 1) ##########")
print(myCorpus.unigramGenerate(1))

print("########## UNIGRAM GENERATE (code 2) ##########")
print(myCorpus.unigramGenerate(2))

print("########## BIGRAM GENERATE ##########")
print(myCorpus.bigramGenerate(1))

print("########## TRIGRAM GENERATE ##########")
print(myCorpus.trigramGenerate(1))