import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
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

trainX, testX, trainy, testy = train_test_split(relative_stat_values, outcomes, test_size=0.005, random_state=2)

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

df_init = pd.read_csv("SVMmarch.csv",delimiter=',')

Team1 = df_init['Team1'].tolist()
Team2 = df_init['Team2'].tolist()

stats = ['FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF']
next_round = []
rounds = []

for round in range(0,7):

    next_roundLR = []

    rounds.append(Team1)
    rounds.append(Team2)

    if round <= 5:
        for i in range(0,len(Team1)):

            team1stat = []
            team2stat = []

            df_stat1 = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{Team1[i]}/{2022}-gamelogs.html')[0]
            df_stat2 = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{Team2[i]}/{2022}-gamelogs.html')[0]


            ppg1 = df_stat1['Unnamed: 5_level_0']['Tm']
            ppg2 = df_stat2['Unnamed: 5_level_0']['Tm']

            points1 = []
            points2 = []

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

            predictLR = modelsvm.predict(input_data)

            if predictLR == 1:

                next_roundLR.append(Team1[i])

            elif predictLR == 0:

                next_roundLR.append(Team2[i])

            else:

                print('Error occurred no valid prediction')
            

        Team1 = []
        Team2 = []

        for i in range(0,len(next_roundLR)):

            if (i % 2) == 0:

                Team1.append(next_roundLR[i])
            else:
                Team2.append(next_roundLR[i])

round64 = []
round32 = []
round16 = []
round8 = []
round4 = []
round2 = []
round1 = []


for i in range(0,len(rounds[0])):

    round64.append(rounds[0][i])
    round64.append(rounds[1][i])
    if i > len(rounds[2]) - 1:
        continue
    else:
        round32.append(rounds[2][i])
        round32.append(rounds[3][i])
    if i > len(rounds[4]) - 1:
        continue
    else:
        round16.append(rounds[4][i])
        round16.append(rounds[5][i])
    if i > len(rounds[6]) - 1:
        continue
    else:
        round8.append(rounds[6][i])
        round8.append(rounds[7][i])
    if i > len(rounds[8]) - 1:
        continue
    else:
        round4.append(rounds[8][i])
        round4.append(rounds[9][i])
    if i > len(rounds[10]) - 1:
        continue
    else:
        round2.append(rounds[10][i])
        round2.append(rounds[11][i])
    if i > len(rounds[12]) - 1:
        continue
    else:
        round1.append(rounds[12][i])


r64 = np.empty_like(np.asarray(round64))
r32 = np.empty_like(np.asarray(round64))
r16 = np.empty_like(np.asarray(round64))
r8 = np.empty_like(np.asarray(round64))
r4 = np.empty_like(np.asarray(round64))
r2 = np.empty_like(np.asarray(round64))
r1 = np.empty_like(np.asarray(round64))

for k in range(0,len(round64)):

    r64[k] = round64[k]

for k in range(0,len(round32)):

    r32[k] = round32[k]

for k in range(0,len(round16)):

    r16[k] = round16[k]

for k in range(0,len(round8)):

    r8[k] = round8[k]

for k in range(0,len(round4)):

    r4[k] = round4[k]

for k in range(0,len(round2)):

    r2[k] = round2[k]

for k in range(0,len(round1)):

    r1[k] = round1[k]


picks = pd.DataFrame({'64':r64,
                    '32':r32,
                    '16':r16,
                    '8':r8,
                    '4':r4,
                    '2':r2,
                    '1':r1})

picks.to_csv('SVMguess.csv')
