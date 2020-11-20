import os
from hmm import HMM
from util import sentence_tokenizer

tokenized_sentences = sentence_tokenizer('./train.txt');

# create a dictionary of all the tags and the hidden observations
tag_var = {}
obs_var = {}

tag_idx = -1
obs_idx = -1

target = []
obs_seq = []

for sentence in tokenized_sentences:
    if(len(sentence) == 0): continue
    for word in sentence:
        if(word[1] not in tag_var):
            tag_idx += 1
            tag_var[word[1]] = tag_idx
        if(word[0] not in obs_var):
            obs_idx += 1
            obs_var[word[0]] = obs_idx
        target.append(tag_var[word[1]])
        obs_seq.append(obs_var[word[0]])

print(len(tag_var),'\n',len(obs_var))
