import sys
from collections import defaultdict
import math
import random
import os
import os.path

def convert_line_to_sequence(line, lexicon=None):
    sequence = line.lower().strip().split()
    if lexicon: 
        return [word if word in lexicon else "UNK" for word in sequence]
    else: 
        return sequence

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                yield convert_line_to_sequence(line, lexicon)

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

def get_ngrams(sequence, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    """

    ngrams = []

    # When unigram, manually add START, as algorithm below skips it
    if n is 1:
        ngrams.append(('START',))

    # Loop through corpus
    endRange = (len(sequence)+1)
    for index_word in range(0, endRange):
        # Range of tuples
        tuple_gram = ()
        for index_gram in range(index_word-n+1, index_word+1):
            word = None
            # figure out word
            if index_gram < 0:
                word = 'START'
            elif index_gram >= len(sequence):
                word = 'STOP'
            else:
                word = sequence[index_gram]
            # constructor tuple
            if word:
                tuple_gram = tuple_gram + (word,)
        
        # append to list
        ngrams.append(tuple_gram)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.word_count = 0.0

        ##Your code here
        for sentence in corpus:
            # Add unigram counts
            for unigram in get_ngrams(sentence, 1):
                self.word_count = self.word_count + 1
                if unigram in self.unigramcounts:
                    self.unigramcounts[unigram] = self.unigramcounts[unigram] + 1
                else:
                    self.unigramcounts[unigram] = 1
            # Add bigram counts
            for bigram in get_ngrams(sentence, 2):
                if bigram in self.bigramcounts:
                    self.bigramcounts[bigram] = self.bigramcounts[bigram] + 1
                else:
                    self.bigramcounts[bigram] = 1
            # Add trigram counts
            for trigram in get_ngrams(sentence, 3):
                if trigram in self.trigramcounts:
                    self.trigramcounts[trigram] = self.trigramcounts[trigram] + 1
                else:
                    self.trigramcounts[trigram] = 1

        return

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """

        tri_count = self.trigramcounts.get(trigram, 0.0)
        bi_count = float(self.bigramcounts.get(trigram[:-1], 0.0))
        
        return (tri_count/bi_count) if bi_count != 0 else 0.0

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """

        bi_count = self.bigramcounts.get(bigram, 0.0)
        uni_count = float(self.unigramcounts.get(bigram[:-1], 0.0))

        return (bi_count/uni_count) if uni_count != 0 else 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 
         
        return self.unigramcounts.get(unigram, 0.0)/self.word_count

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return float(lambda1 * self.raw_trigram_probability(trigram)) + (lambda2 * self.raw_bigram_probability(trigram[-2:])) + (lambda3 * self.raw_unigram_probability(trigram[-1:]))
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        
        trigrams = get_ngrams(sentence, 3)

        logprob = 0.0
        for gram in trigrams:
            smoothedprob = self.smoothed_trigram_probability(gram)
            if smoothedprob == 0.0:
                continue
            p = math.log(smoothedprob, 2)
            logprob = logprob + p


        return float(logprob)

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """

        log_probability = 0.0
        wordcount = 0
        for sentence in corpus:
            wordcount = wordcount + (len(sentence))
            log_probability = log_probability + self.sentence_logprob(sentence)

        return 2**(-float(log_probability/wordcount))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0.0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp2:
                correct = correct+1
            total = total + 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < pp2:
                correct = correct+1
            total = total + 1
        
        return correct/total


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    
    # test code

    # print(get_ngrams(["natural", "language", "processing"],1))
    # print(get_ngrams(["natural", "language", "processing"],2))
    # print(get_ngrams(["natural", "language", "processing"],3))
    #
    # print(len(model.unigramcounts))
    # print(len(model.bigramcounts))
    # print(len(model.trigramcounts))
    #
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])
    #
    # print("Unigram: " + str(model.raw_unigram_probability(('department',))))
    # print("Bigram: " + str(model.raw_bigram_probability(('highway', 'department'))))
    # print("Trigram: " + str(model.raw_trigram_probability(('state','highway','department'))))
    #
    # print("Smoothed Trigram: " + str(model.smoothed_trigram_probability(('state','highway','department'))))
    # print("Sentence Log Probability: " + str(model.sentence_logprob('The State Highway department')))

    #end test code

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print("Train Perplexity: " + str(pp))
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print("Test Perplexity: " + str(pp))


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('./ets_toefl_data/train_high.txt', './ets_toefl_data/train_low.txt', "./ets_toefl_data/test_high", "./ets_toefl_data/test_low")
    # print(acc)

