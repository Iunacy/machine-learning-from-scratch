import numpy as np

probabilities = np.array([0.91, 0.72, 0.44, 0.31, 0.82, 0.15, 0.61, 0.48])
targets       = np.array([1,    1,    1,    0,    0,    0,    1,    0])

threshold = 0.7

def convert(probabilities,threshold):
    predictions=[]
    for probability in probabilities:
        if probability > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def positives_negatives_calc(predictions,targets):
    true_positives=0
    true_negatives=0
    false_positives=0
    false_negatives=0
    for i in range(len(predictions)):
        if predictions[i]==targets[i]==1:
            true_positives+=1
        elif predictions[i]==targets[i]==0:
            true_negatives+=1
        elif predictions[i]==1 and targets[i] ==0:
            false_positives+=1
        else:
            false_negatives+=1
    return true_positives,true_negatives,false_positives,false_negatives

def precision_recall_f1score_calc(true_positives,true_negatives,false_positives,false_negatives):
    precision=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)
    f1score=2/(1/precision)+(1/recall)
    return precision,recall,f1score

predictions = convert(probabilities, threshold)

tp, tn, fp, fn = positives_negatives_calc(predictions, targets)

precision, recall, f1score = precision_recall_f1score_calc(tp, tn, fp, fn)

print("Predictions:", predictions)
print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1score)

