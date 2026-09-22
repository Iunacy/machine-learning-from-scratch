import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()

x=data['data']
y=data['target']

weights=np.zeros(x.shape[1])
bias=0
learning_rate=0.01

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

means=np.mean(x_train,axis=0)
stds=np.std(x_train,axis=0)

features_train_scaled=(x_train-means)/stds
features_test_scaled=(x_test-means)/stds





def logits_calc(features,weights,bias):
    return features @ weights + bias

def sigmoid(logits):
    return 1/(1+np.exp(-logits))

def loss_calc(probabilities,targets):
    return -(targets*np.log(probabilities)+(1-targets)*np.log(1-probabilities))

def binary_converter(probabilities):
    binary_list=[]
    for probability in probabilities:
        compare=probability>=0.5
        binary=compare.astype(int)
        binary_list.append(binary)
    return binary_list


for epoch in range(1000):
    logits=logits_calc(features_train_scaled,weights,bias)
    probabilities=sigmoid(logits)
    loss=loss_calc(probabilities,y_train)
    errors=probabilities-y_train
    weight_gradients=(features_train_scaled.T @ errors)/len(features_train_scaled)
    bias_gradient=np.mean(errors)
    weights-=learning_rate*weight_gradients
    bias-=learning_rate*bias_gradient

test_logits=logits_calc(features_test_scaled,weights,bias)
test_probabilities=sigmoid(test_logits)
test_predictions=binary_converter(test_probabilities)

print(bias.shape)



