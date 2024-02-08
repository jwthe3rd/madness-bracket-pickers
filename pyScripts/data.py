import pandas as pd
import numpy as np

def split(word):
    return [char for char in word]

namesData = pd.read_csv('./names.csv',delimiter=',')

bigstat_team = []

common_names = namesData['Common Name'].tolist()
site_names = namesData['site name'].tolist()

stats = ['FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF']

opponent_3PA = []
team_3PA = []
outcome = []

years = years = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]

for l in years:
    print(l)
    done = []
    for k in site_names:
        year = l
        team = k
        done.append(common_names[site_names.index(k)])

        df = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{team}/{year}-gamelogs.html')[0]
        opponents = df['Unnamed: 3_level_0']['Opp'].tolist()
        threePA = df['Unnamed: 5_level_0']['Tm']
        threePAschool = []
        for i in threePA:
            try:
                threePAschool.append(int(i))
            except ValueError:
                continue
        avg3PA = sum(threePAschool)/len(threePAschool)
        stat_array_team = []
        
        for stat in stats:

            temp_array = []
            stat_array = df['School'][stat]

            for value in stat_array:
                try:
                    temp_array.append(float(value))
                except ValueError:
                    continue
            stat_array_team.append(sum(temp_array)/len(temp_array))
        

        for i in opponents:
            if i not in done:
                if i in common_names:
                    dfOpp = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{site_names[common_names.index(i)]}/{year}-gamelogs.html')[0]
                    threePAOpp = dfOpp['Unnamed: 5_level_0']['Tm']
                    threePAschool = []
                    for j in threePAOpp:
                        try:
                            threePAschool.append(int(j))
                        except ValueError:
                            continue
                    avg3PAOpp = sum(threePAschool)/len(threePAschool)

                    opponent_3PA.append(avg3PAOpp)
                    team_3PA.append(avg3PA)

                    result = df['Unnamed: 4_level_0']['W/L'][opponents.index(i)]
                    resultChar = split(result)

                    if resultChar[0] == 'W':
                        outcome.append(1)
                    else:
                        outcome.append(0)
            else:
                continue

print(outcome)