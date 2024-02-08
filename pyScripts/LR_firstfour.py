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
from sklearn import preprocessing
import csv

oppStats = ['PPG opp','avg FG made opp','avg FG attempts opp','Avg 3P made opp','avg 3P attempts opp','avg FT made opp','avg FT attempts opp','avg Off Reb opp','Avg Team Reb opp','Avg Assists opp','Avg Steals opp','Avg Blocks opp','Avg turnovers opp','Avg Pers Fouls opp']
teamStats = ['PPG','avg FG made','avg FG attempts','Avg 3P made','avg 3P attempts','avg FT made','avg FT attempts','avg Off Reb','Avg Team Reb','Avg Assists','Avg Steals','Avg Blocks','Avg turnovers','Avg Pers Fouls']
dF = pd.read_csv("dataPower5.csv",delimiter=',')

outcomes = dF['Outcome']
relative_stat_values = []
for i in teamStats:
    arr = np.asarray(dF[i])/np.asarray(dF[oppStats[teamStats.index(i)]])
    relative_stat_values.append(arr)

relative_stat_values = np.asarray(relative_stat_values).T

trainX, testX, trainy, testy = train_test_split(relative_stat_values, outcomes, test_size=0.005, random_state=2)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
modelLR = LogisticRegression(solver='lbfgs')
modelLR.fit(trainX, trainy)
# predict probabilities
lr_probs = modelLR.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

y_pred = modelLR.predict(testX)

cnf_matrix = metrics.confusion_matrix(testy, y_pred)
print(cnf_matrix)


# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model

df_init = pd.read_csv("firstfour.csv",delimiter=',')

Team1 = df_init['Team1'].tolist()
Team2 = df_init['Team2'].tolist()

stats = ['FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF']

rounds = []

print(Team1)
print(Team2)

for i in range(0,len(Team1)):

    next_round = []
    team1stat = []
    team2stat = []

    df_stat1 = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{Team1[i]}/{2022}-gamelogs.html')[0]
    df_stat2 = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{Team2[i]}/{2022}-gamelogs.html')[0]


    ppg1 = df_stat1['Unnamed: 5_level_0']['Tm']
    ppg2 = df_stat2['Unnamed: 5_level_0']['Tm']

    points1 = []
    points2 = []

    print(ppg1)
    print(ppg2)

    for value in ppg1:
        try:
            points1.append(int(value))
        except ValueError:
            continue

    for value in ppg2:
        try:
            points2.append(int(value))
        except ValueError:
            continue

    print(points1)
    print(points2)

    team1stat.append(sum(points1)/len(points1))
    team2stat.append(sum(points2)/len(points2))

    for stat in stats:

        temp_array = []
        temp_array2 = []
        stat_array1 = df_stat1['School'][stat]
        stat_array2 = df_stat2['School'][stat]

        for value in stat_array1:
            try:
                temp_array.append(float(value))
            except ValueError:
                continue

        for value in stat_array2:
            try:
                temp_array2.append(float(value))
            except ValueError:
                continue

        team1stat.append(sum(temp_array)/len(temp_array))
        team2stat.append(sum(temp_array2)/len(temp_array2))


    input_data = []

    for q in range(0,len(team1stat)):

        input_data.append(team1stat[q]/team2stat[q])

    input_data = np.asarray(input_data).reshape(1,-1)
    #temp = np.array(input_data).reshape((len(input_data), 1))

    predictLR = modelLR.predict(input_data)

    if predictLR == 1:

        next_round.append(Team1[i])

    elif predictLR == 0:

        next_round.append(Team2[i])

    else:

        print('Error occurred no valid prediction')

    rounds.append(next_round)



print(rounds)