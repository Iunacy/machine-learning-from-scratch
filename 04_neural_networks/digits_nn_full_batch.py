from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

digits=load_digits()

features=digits.data/16
targets=digits.target

x_batch,x_test,targets,y_test=train_test_split(features,targets,test_size=0.2,random_state=42,stratify=targets)

rng=np.random.default_rng(42)
weights_1=rng.normal(0,0.1,size=(64,128))
bias_1=np.zeros(128)

weights_2=rng.normal(0,0.1,size=(128,10))
bias_2=np.zeros(10)

learning_rate=0.01

def relu(logits):
    return np.maximum(0,logits)

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    logits_exp=np.exp(shifted_logits)
    return logits_exp/np.sum(logits_exp,axis=1,keepdims=True)

def loss_calc(probabilities,targets):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    return -np.mean(np.log(correct_probabilities))

def mistakes_finder(predictions,targets):
    mistake={}
    for i in range(len(predictions)):
        if predictions[i]!=targets[i]:
            mistake[i]={'Prediction':int(predictions[i]),'Target':int(targets[i])}
    return mistake

for epoch in range(10000):
    hidden_logits=x_batch @ weights_1 + bias_1
    hidden_activations=relu(hidden_logits)
    output_logits=hidden_activations @ weights_2 + bias_2
    probabilities=softmax(output_logits)

    output_errors=probabilities.copy()
    output_errors[np.arange(len(targets)),targets]-=1
    output_errors/=len(x_batch)

    weights_2_gradients=hidden_activations.T @ output_errors
    bias_2_gradients=np.sum(output_errors,axis=0)

    hidden_errors=output_errors @ weights_2.T
    hidden_errors[hidden_logits<=0]=0

    weights_1_gradients=x_batch.T @ hidden_errors
    bias_1_gradients=np.sum(hidden_errors,axis=0)

    weights_1-=learning_rate * weights_1_gradients
    bias_1-=learning_rate * bias_1_gradients
    weights_2-=learning_rate * weights_2_gradients
    bias_2-=learning_rate * bias_2_gradients

    loss=loss_calc(probabilities,targets)

    if epoch % 10 == 0:
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == targets)
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")

hidden_test = relu(x_test @ weights_1 + bias_1)
test_logits = hidden_test @ weights_2 + bias_2
test_predictions = np.argmax(test_logits, axis=1)

print("Test accuracy:", np.mean(test_predictions == y_test))
mistakes=mistakes_finder(test_predictions,y_test)
print(mistakes)


i = 2
plt.imshow(x_test[i].reshape(8, 8), cmap="gray")
plt.title(f"Predicted: {test_predictions[i]} | Actual: {y_test[i]}")
plt.axis("off")
plt.show()

test_probabilities = softmax(test_logits)

top_3 = np.argsort(test_probabilities[2])[-3:][::-1]

for digit in top_3:
    print(f"Digit {digit}: {test_probabilities[2, digit]:.2%}")