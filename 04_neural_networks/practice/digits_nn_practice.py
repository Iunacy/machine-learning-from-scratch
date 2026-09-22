import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits=load_digits()
x=digits.data
y=digits.target

x=x/16

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

rng=np.random.default_rng(42)

weights_1=rng.normal(0,0.1,size=(x_train.shape[1],128))
weights_2=rng.normal(0,0.1,size=(128,10))
bias_1=np.zeros(128)
bias_2=np.zeros(10)

learning_rate=0.1
batch_size=64

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    logits_exp=np.exp(shifted_logits)
    return logits_exp/np.sum(logits_exp,axis=1,keepdims=True)

def relu(logits):
    return np.maximum(0,logits)

def loss_calc(probabilities,targets):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    return -np.mean(np.log(correct_probabilities))

def logits_calc(x,weights,bias):
    return x @ weights + bias

for epoch in range(1000):
    indices=rng.permutation(len(x_train))
    x_shuffled=x_train[indices]
    y_shuffled=y_train[indices]
    for start in range(0,len(x_train),batch_size):
        end=start+batch_size
        x_batch=x_shuffled[start:end]
        y_batch=y_shuffled[start:end]

        hidden_logits=logits_calc(x_batch,weights_1,bias_1)
        hidden_activate=relu(hidden_logits)
        output_logits=logits_calc(hidden_activate,weights_2,bias_2)
        probabilities=softmax(output_logits)

        loss=loss_calc(probabilities,y_batch)

        output_error=probabilities.copy()
        output_error[np.arange(len(y_batch)),y_batch]-=1
        output_error/=len(x_batch)

        weights_2_gradients=hidden_activate.T @ output_error
        bias_2_gradients=np.sum(output_error,axis=0)

        hidden_error=output_error @ weights_2.T
        hidden_error[hidden_logits<=0]=0

        weights_1_gradients=x_batch.T @ hidden_error
        bias_1_gradients=np.sum(hidden_error,axis=0)

        weights_1-=learning_rate * weights_1_gradients
        bias_1-=learning_rate * bias_1_gradients
        weights_2-=learning_rate * weights_2_gradients
        bias_2-=learning_rate * bias_2_gradients

    if epoch%100==0:
        print('Epoch:',epoch,'Loss:',loss)

test_logits=x_test @ weights_1 + bias_1
test_activate=np.maximum(0,test_logits)
output_test_logits=test_activate @ weights_2 + bias_2
test_probabilities=softmax(output_test_logits)

test_predictions=np.argmax(test_probabilities,axis=1)
print('Test accuracy:',np.mean(test_predictions==y_test))

