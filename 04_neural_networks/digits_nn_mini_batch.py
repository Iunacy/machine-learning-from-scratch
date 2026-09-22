import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits=load_digits()
X=digits.data
y=digits.target
X=X/16.0

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

rng=np.random.default_rng(42)

input_size=x_train.shape[1]
hidden_size=128
output_size=10

weights_1=rng.normal(0,0.1,size=(input_size,hidden_size))
bias_1=np.zeros(hidden_size)

weights_2=rng.normal(0,0.1,size=(hidden_size,output_size))
bias_2=np.zeros(output_size)

learning_rate=0.1
epochs=100000
batch_size=64

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    exp_logits=np.exp(shifted_logits)
    probabilities=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
    return probabilities



batch_size=64
for epoch in range(epochs):
    indices=rng.permutation(len(x_train))
    x_shuffled=x_train[indices]
    y_shuffled=y_train[indices]
    for start in range(0, len(x_train), batch_size):
        end=start+batch_size
        x_batch=x_shuffled[start:end]
        y_batch=y_shuffled[start:end]

        hidden_logits=x_batch @ weights_1 + bias_1
        hidden_activations=np.maximum(0,hidden_logits)
        output_logits=hidden_activations @ weights_2 +bias_2 
        probabilities=softmax(output_logits)

        loss=-np.mean(np.log(probabilities[np.arange(len(y_batch)),y_batch]))

        output_errors=probabilities.copy()
        output_errors[np.arange(len(y_batch)),y_batch]-=1
        output_errors/=len(x_batch)

        weights_2_gradients=(hidden_activations.T @ output_errors)
        bias_2_gradients=np.sum(output_errors,axis=0)

        hidden_errors=output_errors @ weights_2.T
        hidden_errors[hidden_logits<=0]=0

        weights_1_gradients=x_batch.T @ hidden_errors
        bias_1_gradients=np.sum(hidden_errors,axis=0)

        weights_1-=learning_rate * weights_1_gradients
        bias_1-=learning_rate * bias_1_gradients

        weights_2-=learning_rate * weights_2_gradients
        bias_2-=learning_rate * bias_2_gradients

    if epoch%10==0:
        print('Epoch:',epoch,'Loss:',loss)

test_logits=x_test @ weights_1 + bias_1
test_activate=np.maximum(0,test_logits)
output_test_logits=test_activate @ weights_2 + bias_2
test_probabilities=softmax(output_test_logits)

test_predictions=np.argmax(test_probabilities,axis=1)
print('Test accuracy:',np.mean(test_predictions==y_test))


        
