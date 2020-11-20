import os
from hmm import HMM
from util import sentence_tokenizer

tokenized_sentences = sentence_tokenizer('./test.txt');

# create a dictionary of all the tags and the hidden observations
obs_var = {}
obs_idx = 0 

for sentence in tokenized_sentences:
    if(len(sentence) == 0): continue
    for word in sentence:
       if(word[0] not in obs_var):
            obs_idx += 1
            obs_var[word[0]] = obs_idx
        obs_seq.append(obs_var[word[0]])

print(tag_var)
print(obs_var)

# Setting model
#hmm = HMM(len(tag_var), len(obs_var))
#hmm.train(obs_seq, verbose=1)
#hmm.show_model()

#Calculating the viterbi_path for the testing data based on the model trained
viterbi_path, hits, words = hmm.given(obs_seq)['viterbi']
#print('\n'.join([hidden_var_name[h_id] for h_id in viterby_path]))

accuracy = (hits / words) * 100
print('Accuracy: ', accuracy)
