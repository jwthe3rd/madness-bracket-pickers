import pandas as pd
import numpy as np
import random


teams = pd.read_csv('randommarch.csv')

team1 = teams['Team1']
team2 = teams['Team2']

teams1 = teams['Team1']
teams2 = teams['Team2']

rounds = []

for j in range(0,7):

    next_round = []

    if j <= 5:

        for i in range(0,len(team1)):

            k = random.randint(0,1)

            if k == 1:
                next_round.append(team1[i])
            elif k == 0:
                next_round.append(team2[i])

            else:
                print('problem wit rand int')


        rounds.append(next_round)

        team1 = []
        team2 = []

        for i in range(0,len(next_round)):

            if (i % 2) == 0:

                team1.append(next_round[i])
            else:
                team2.append(next_round[i])

print(next_round)
r64 = []
r32 = []
r16 = []
r8 = []
r4 = []
r2 = []
r1 = []


for p in teams1:
    r64.append(p)
for p in teams2:
    r64.append(p)
for p in rounds[0]:
    r32.append(p)
for p in rounds[1]:
    r16.append(p)
for p in rounds[2]:
    r8.append(p)
for p in rounds[3]:
    r4.append(p)
for p in rounds[4]:
    r2.append(p)
for p in rounds[5]:
    r1.append(p)


for i in range(0,len(r64)+1):

    print(i)

    if i > len(r32):
        r32.append(0)
    if i > len(r16):
        r16.append(0)
    if i > len(r8):
        r8.append(0)
    if i > len(r4):
        r4.append(0)
    if i > len(r2):
        r2.append(0)
    if i > len(r1):
        r1.append(0)

print(len(r1))
picks = pd.DataFrame({'64':r64,
                    '32':r32,
                    '16':r16,
                    '8':r8,
                    '4':r4,
                    '2':r2,
                    '1':r1})

picks.to_csv('randomguess.csv')


