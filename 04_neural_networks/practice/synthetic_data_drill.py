import numpy as np

x_batch = np.random.randn(256, 784)
targets = np.random.randint(0, 10, size=256)
learning_rate = 0.01
epochs = 100

rng=np.random.default_rng(42)

weights_1=rng.normal(0,0.1,size=(784,128))
bias_1=np.zeros(128)

weights_2=rng.normal(0,0.1,size=(128,10))
bias_2=np.zeros(10)

def relu(logits):
    return np.maximum(0,logits)

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    logits_exp=np.exp(shifted_logits)
    probabilities=logits_exp/np.sum(logits_exp,axis=1,keepdims=True)
    return probabilities

def loss_calc(probabilities):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    loss=-np.log(correct_probabilities)
    return np.mean(loss)

def logits_calc(x,weights,bias):
    return x @ weights + bias

for epoch in range(epochs):
    hidden_logits=logits_calc(x_batch,weights_1,bias_1)
    hidden_activations=relu(hidden_logits)

    output_logits=logits_calc(hidden_activations,weights_2,bias_2)
    probabilities=softmax(output_logits)

    loss=loss_calc(probabilities)

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

    if epoch % 10 == 0:
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == targets)
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")

x_test = rng.normal(size=(256, 784))
y_test = rng.integers(0, 10, size=256)

hidden_test = relu(x_test @ weights_1 + bias_1)
test_logits = hidden_test @ weights_2 + bias_2
test_predictions = np.argmax(test_logits, axis=1)

print("Test accuracy:", np.mean(test_predictions == y_test))