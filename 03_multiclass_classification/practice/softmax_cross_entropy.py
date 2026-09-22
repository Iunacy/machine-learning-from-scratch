import numpy as np

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    exp_logits=np.exp(shifted_logits)
    probabilities=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
    return probabilities

def loss_calc(probabilities,targets):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    losses=-np.log(correct_probabilities)
    return np.mean(losses)
