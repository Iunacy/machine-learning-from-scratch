import numpy as np

targets     = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2])
predictions = np.array([0, 1, 1, 2, 1, 2, 2, 0, 0, 2])

num_classes = 3

confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

def precision_recall_f1score_calc(true_positives,false_positives,false_negatives):
    precision=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)
    f1score=2/(1/precision+1/recall)
    return precision,recall,f1score



for actual, predicted in zip(targets, predictions):
    confusion_matrix[actual,predicted]+=1

total_TP = 0
total_FP = 0
total_FN = 0
f1_score_list=[]
supports=[]
for c in range(0,num_classes):
    TP=confusion_matrix[c,c]
    FP=np.sum(confusion_matrix[:,c])-TP
    FN=np.sum(confusion_matrix[c,:])-TP
    

    precision,recall,f1_score=precision_recall_f1score_calc(TP,FP,FN)

    support=np.sum(confusion_matrix[c,:])
    supports.append(support)
    total_TP+=TP
    total_FP+=FP
    total_FN+=FN

    f1_score_list.append(f1_score)
    
    print('Class:',c,'Precision:',precision,'Recall:',recall,'F1 Score:',f1_score)
macro_f1=np.mean(f1_score_list)
print('Macro_F1:',macro_f1)
print('Weighted average:',(np.array(f1_score_list) @ np.array(supports))/len(targets))
micro_precision,micro_recall,micro_f1_score=precision_recall_f1score_calc(total_TP,total_FP,total_FN)
print('Precision:',micro_precision,'Recall:',micro_recall,'F1 Score:',micro_f1_score)