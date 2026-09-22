import numpy as np

logits = np.array([
    [2.0, 1.0, 0.0],
    [0.5, 1.5, -0.5],
    [-1.0, 2.0, 1.0]
])

targets = np.array([0, 0, 2])


def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    logits_exp=np.exp(shifted_logits)
    probabilities=logits_exp/np.sum(logits_exp,axis=1,keepdims=True)
    return probabilities

def loss_calc(probabilities,targets):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    loss=-np.log(correct_probabilities)
    return np.mean(loss)

def predict(probabilities):
    return np.argmax(probabilities)

probabilities = softmax(logits)
loss = loss_calc(probabilities, targets)
predictions = np.argmax(probabilities, axis=1)

print("Probabilities:\n", probabilities)
print("Row sums:", np.sum(probabilities, axis=1))
print("Predictions:", predictions)
print("Mean loss:", loss)