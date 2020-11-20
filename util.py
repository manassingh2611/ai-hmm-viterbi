import math
import re
import numpy as np
from functools import reduce

EPS = 1e-4
ZERO = 1e-300
LOG_ZERO = math.log(ZERO)

def normalize2d(vec):
    return (vec.transpose() / vec.sum(axis=1)).transpose()

def normalize1d(vec):
    return vec / vec.sum()

def log_mul(*arrays):
    return np.array(arrays).sum(axis=0)

log_vec_mul = np.vectorize(log_mul)

def log_div(*arrays):
    return arrays[0] - log_mul(*arrays[1:])

log_vec_div = np.vectorize(log_div)

def log_add(x, y):
    a = max(x, y)
    b = min(x, y)
    return a + math.log(1.0 + math.exp(b-a))

log_vec_add = np.vectorize(log_add)

def log_sum(*arrays):
    return reduce(lambda x,y: log_add(x, y), arrays)

#input sanitization
# return type list(tuple(word, tag))
def sentence_tokenizer(filepath):
    sentences = []
    with open(filepath, 'r') as _:
        sentence = []
        for line in _:
            line = line.strip()
            if line:
                line = line.split()
                if(re.match(r"^\w+$", line[0]) == None): continue
                line[0] = line[0].lower()
                sentence.append(tuple(line[:2]))
            else:
                sentences.append(sentence)
                sentence = []
    return sentences

def word_tag_counts(tokenized_sentences):  
    count_word_tag = {} #frequency of word-tag pairs
    count_tags = {}     #frequency of tags  
    for sentence in tokenized_sentences:
        for pair in sentence:
            if(pair[1] in count_tags):
                count_tags[pair[1]] += 1
            else:
                count_tags[pair[1]] = 1
            if(pair in count_word_tag):
                count_word_tag[pair] += 1
            else:
                count_word_tag[pair] = 1
    return (count_word_tag, count_tags)

def tag_pair_frequencies(tokenized_sentences):
    tag_pair_frequency = {}
    for sentence in tokenized_sentences:
        if(len(sentence) == 0): continue
        tag_pair_frequency[(sentence[0][1], '<S>')] = tag_pair_frequency.get((sentence[0][1], '<S>'), 0) + 1
        tag_pair_frequency[('<E>', sentence[len(sentence) - 1][1])] = tag_pair_frequency.get(('<E>', sentence[len(sentence) - 1][1]), 0) + 1
        for i in range(len(sentence) - 1):
            tag_pair = (sentence[i + 1][1], sentence[i][1])
            tag_pair_frequency[tag_pair] = tag_pair_frequency.get(tag_pair, 0) + 1
    return tag_pair_frequency

def tag_counts(tokenized_sentences):
    tags = {}
    total_tags = 0
    for sentence in tokenized_sentences:
        total_tags += len(sentence)
        if(len(sentence) == 0): continue
        tags[sentence[0][1]] = tags.get(sentence[0][1], 0) + 1
        for wt in sentence:
            tags[wt[1]] = tags.get(wt[1], 0)
    return (tags, total_tags)
