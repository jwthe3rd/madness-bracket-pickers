import re
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn import svm

oppStats = ['PPG opp','avg FG made opp','avg FG attempts opp','Avg 3P made opp','avg 3P attempts opp','avg FT made opp','avg FT attempts opp','avg Off Reb opp','Avg Team Reb opp','Avg Assists opp','Avg Steals opp','Avg Blocks opp','Avg turnovers opp','Avg Pers Fouls opp']
teamStats = ['PPG','avg FG made','avg FG attempts','Avg 3P made','avg 3P attempts','avg FT made','avg FT attempts','avg Off Reb','Avg Team Reb','Avg Assists','Avg Steals','Avg Blocks','Avg turnovers','Avg Pers Fouls']
dF = pd.read_csv("dataPower5.csv",delimiter=',')

outcomes = dF['Outcome']
relative_stat_values = []
for i in teamStats:
    arr = np.asarray(dF[i])/np.asarray(dF[oppStats[teamStats.index(i)]])
    relative_stat_values.append(arr)

relative_stat_values = np.asarray(relative_stat_values).T

trainX, testX, trainy, testy = train_test_split(relative_stat_values, outcomes, test_size=0.5, random_state=2)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

y_pred = model.predict(testX)

cnf_matrix = metrics.confusion_matrix(testy, y_pred)
print(cnf_matrix)


# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model

modelsvm = svm.SVC()
modelsvm.fit(trainX,trainy)

y_predSVM = modelsvm.predict(testX)

hit = 0

print(y_predSVM)
print(testy.tolist())
testylist = testy.tolist()

for i in range(0,len(y_predSVM)):
    if y_predSVM[i]==testylist[i]:
        hit = hit + 1

print(f'Support Vector Machine Accuracy: {100*hit/len(y_predSVM)} %')

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()





