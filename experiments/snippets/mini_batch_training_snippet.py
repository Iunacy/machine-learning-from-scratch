def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    return np.exp(shifted_logits)/np.sum(np.exp(shifted_logits),axis=1,keepdims=True)

def relu(logits):
    return np.maximum(0,logits)

batch_size=64
for epoch in range(100):
    indices=rng.permutation(len(x_train))
    x_shuffled=x_train[indices]
    y_shuffled=y_train[indices]
    for start in range(0,len(x_train),batch_size):
        end=start+batch_size
        x_batch=x_shuffled[start:end]
        y_batch=y.shuffled[start:end]

        hidden_logits=x_batch @ weights_1 +bias_1
        hidden_activations=relu(hidden_logits)
        output_logits=hidden_activations @ weights_2 + bias_2 
        probabilities=softmax(output_logits)

        loss=-np.mean(np.log(probabilities[np.arange(len(y_batch)),y_batch]))

        output_error=probabilities.copy()
        output_error[np.arange(len(y_batch)),y_batch]-=1
        output_error/=len(x_batch)

        weights_2_gradients=hidden_activations.T @ output_error
        bias_2_gradients=np.sum(output_error,axis=0)

        hidden_error=output_error @ weights_2.T
        hidden_error[hidden_logits<=0]=0

        weights_1_gradients=x_batch.T @ hidden_error
        bias_1_gradients=np.sum(hidden_error,axis=0)

        weights_1-=learning_rate * weights_1_gradients
        bias_1-=learning_rate * bias_1_gradients

        weights_2-=learning_rate * weights_2_gradients
        bias_2-=learning_rate * bias_2_gradients

