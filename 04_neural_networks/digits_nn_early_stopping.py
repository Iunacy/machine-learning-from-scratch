import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits=load_digits()

x=digits.data
y=digits.target

x=x/16

x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=123,stratify=y_temp)

rng=np.random.default_rng(123)

input_size = 64
hidden_size = 48
output_size = 10
weights_1=rng.normal(0,0.1,size=(input_size,hidden_size))
bias_1=np.zeros(hidden_size)

weights_2=rng.normal(0,0.1,size=(hidden_size,output_size))
bias_2=np.zeros(output_size)

learning_rate=0.01

def classification_report(confusion_matrix,predictions,targets):
    f1_scores=[]
    report={}
    total_tp=0
    total_fp=0
    total_fn=0
    supports=[]
    for i in range(10):
        tp, fp, fn = tp_fp_fn_calc(confusion_matrix, i)
        total_tp+=tp
        total_fn+=fn
        total_fp+=fp
        precision, recall, f1 = precision_recall_f1score_calc(tp, fp, fn)
        f1_scores.append(f1)
        support=np.sum(confusion_matrix[i,:])
        supports.append(support)

        report[i]={
                'support':int(support),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    macro_f1=np.mean(f1_scores)
    micro_precision,micro_recall,micro_f1=precision_recall_f1score_calc(total_tp,total_fp,total_fn)
    weighted_f1=(np.array(supports) @ np.array(f1_scores))/np.sum(supports)
    accuracy=np.mean(predictions==targets)
        

    

    overall = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1)
    }
    return report,overall
    

def tp_fp_fn_calc(confusion_matrix,class_index):
    tp=np.sum(confusion_matrix[class_index,class_index])

    fp=np.sum(confusion_matrix[:,class_index])-tp
    fn=np.sum(confusion_matrix[class_index,:])-tp
    return tp,fp,fn
def precision_recall_f1score_calc(tp,fp,fn):
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision,recall,f1

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    shift_x=x-np.max(x,axis=1,keepdims=True)
    x_exp=np.exp(shift_x)
    probabilities=x_exp/np.sum(x_exp,axis=1,keepdims=True)
    return probabilities
def logits_calc(x,weights,bias):
    return x @ weights + bias

def loss_calc(probabilities,targets):
    correct_probs=probabilities[np.arange(len(targets)),targets]
    return -np.mean(np.log(correct_probs))
min_delta=0.001
best_val_loss = np.inf
patience = 10
reference_val_loss=np.inf
patience_counter = 0
batch_size=64
train_losses=[]
val_losses=[]
for epoch in range(2000):
    indices=rng.permutation(len(x_train))

    x_shuffled=x_train[indices]
    y_shuffled=y_train[indices]
    for start in range(0,len(x_train),batch_size):
        end=start+batch_size
        x_batch=x_shuffled[start:end]
        y_batch=y_shuffled[start:end]

        hl=logits_calc(x_batch,weights_1,bias_1)
        h_a=relu(hl)
        output=logits_calc(h_a,weights_2,bias_2)

        probabilities=softmax(output)

        d_o=probabilities.copy()
        d_o[np.arange(len(y_batch)),y_batch]-=1
        d_o/=len(y_batch)

        loss=loss_calc(probabilities,y_batch)

        weights_2_gradients=h_a.T @ d_o
        bias_2_gradients=np.sum(d_o,axis=0)

        d_h=d_o @ weights_2.T
        d_h[hl<=0]=0

        weights_1_gradients=x_batch.T @ d_h
        bias_1_gradients=np.sum(d_h,axis=0)

        weights_1 -= learning_rate * weights_1_gradients
        bias_1 -= learning_rate * bias_1_gradients

        weights_2 -= learning_rate * weights_2_gradients
        bias_2 -= learning_rate * bias_2_gradients

        
    train_logits=logits_calc(x_train,weights_1,bias_1)
    train_hidden=relu(train_logits)
    output_train=logits_calc(train_hidden,weights_2,bias_2)
    train_probabilities=softmax(output_train)

    train_loss=loss_calc(train_probabilities,y_train)
        

    val_logits=logits_calc(x_val,weights_1,bias_1)
    val_hidden=relu(val_logits)
    output_val_logits=logits_calc(val_hidden,weights_2,bias_2)
    val_predictions=np.argmax(output_val_logits,axis=1)

    val_probabilities = softmax(output_val_logits)
    val_loss = loss_calc(val_probabilities, y_val)

    val_accuracy=np.mean(val_predictions==y_val)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_weights_1 = weights_1.copy()
        best_weights_2 = weights_2.copy()
        best_bias_1 = bias_1.copy()
        best_bias_2 = bias_2.copy()
        
    if val_loss < reference_val_loss - min_delta:
        reference_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter>=patience:
        print("Early stopping at epoch:", epoch)
        break


    
weights_1 = best_weights_1
bias_1 = best_bias_1
weights_2 = best_weights_2
bias_2 = best_bias_2


test_logits=logits_calc(x_test,weights_1,bias_1)
test_activations=relu(test_logits)
output_test_logits=logits_calc(test_activations,weights_2,bias_2)

test_probabilities=softmax(output_test_logits)
predictions=np.argmax(test_probabilities,axis=1)

accuracy=np.mean(predictions==y_test)

confusion_matrix=np.zeros((10,10),dtype=int)

for actual,prediction in zip(y_test,predictions):
    confusion_matrix[actual,prediction]+=1

print(confusion_matrix)


wrong_mask = predictions != y_test

wrong_images = x_test[wrong_mask]
wrong_actual = y_test[wrong_mask]
wrong_predictions = predictions[wrong_mask]

print("Number of mistakes:", len(wrong_images))
print("Actual:", wrong_actual)
print("Predicted:", wrong_predictions)

error_counts = np.zeros(10, dtype=int)

for actual in wrong_actual:
    error_counts[actual] += 1

print(error_counts)

report,overall = classification_report(confusion_matrix,predictions,y_test)

for class_index, metrics in report.items():
    print(
        f"Class {class_index}: "
        f"Support={metrics['support']} "
        f"TP={metrics['tp']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"Precision={metrics['precision']:.3f} "
        f"Recall={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f}"
    )

print("\nOverall:")
print("Accuracy:", overall["accuracy"])
print("Macro F1:", overall["macro_f1"])
print("Weighted F1:", overall["weighted_f1"])
print("Micro F1:", overall["micro_f1"])

plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.show()