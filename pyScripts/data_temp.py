import pandas as pd
import numpy as np

def split(word):
    return [char for char in word]

namesData = pd.read_csv('./names.csv',delimiter=',')

bigstat_team = []

common_names = namesData['Common Name'].tolist()
site_names = namesData['site name'].tolist()

stats = ['FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF']

oppStat= []
teamStat = []
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

        stat_array_team.append(avg3PA)
        
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

                    stat_array_opp = []

                    stat_array_opp.append(avg3PAOpp)
        
                    for stat in stats:

                        temp_array = []
                        stat_array = dfOpp['School'][stat]

                        for value in stat_array:
                            try:
                                temp_array.append(float(value))
                            except ValueError:
                                continue
                        stat_array_opp.append(sum(temp_array)/len(temp_array))

                    result = df['Unnamed: 4_level_0']['W/L'][opponents.index(i)]
                    resultChar = split(result)

                    if resultChar[0] == 'W':
                        outcome.append(1)
                    else:
                        outcome.append(0)

                    teamStat.append(stat_array_team)
                    oppStat.append(stat_array_opp)
            else:
                continue


#'FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF'

avgP = []
FG = []
FGA = []
threeP = []
threePA = []
FT = []
FTA = []
ORB = []
TRB = []
AST = []
STL = []
BLK = []
TOV = []
PF = []

avgPo = []
FGo = []
FGAo = []
threePo = []
threePAo = []
FTo = []
FTAo = []
ORBo = []
TRBo = []
ASTo = []
STLo = []
BLKo = []
TOVo = []
PFo = []

for array in teamStat:
    avgP.append(array[0])
    FG.append(array[1])
    FGA.append(array[2])
    threeP.append(array[3])
    threePA.append(array[4])
    FT.append(array[5])
    FTA.append(array[6])
    ORB.append(array[7])
    TRB.append(array[8])
    AST.append(array[9])
    STL.append(array[10])
    BLK.append(array[11])
    TOV.append(array[12])
    PF.append(array[13])

for array in oppStat:
    avgPo.append(array[0])
    FGo.append(array[1])
    FGAo.append(array[2])
    threePo.append(array[3])
    threePAo.append(array[4])
    FTo.append(array[5])
    FTAo.append(array[6])
    ORBo.append(array[7])
    TRBo.append(array[8])
    ASTo.append(array[9])
    STLo.append(array[10])
    BLKo.append(array[11])
    TOVo.append(array[12])
    PFo.append(array[13])


finalDataF = pd.DataFrame({'Outcome':outcome,
                            'PPG':avgP,
                            'avg FG made':FG,
                            'avg FG attempts':FGA,
                            'Avg 3P made':threeP,
                            'avg 3P attempts':threePA,
                            'avg FT made':FT,
                            'avg FT attempts':FTA,
                            'avg Off Reb':ORB,
                            'Avg Team Reb':TRB,
                            'Avg Assists':AST,
                            'Avg Steals':STL,
                            'Avg Blocks':BLK,
                            'Avg turnovers':TOV,
                            'Avg Pers Fouls':PF,
                            'PPG opp':avgPo,
                            'avg FG made opp':FGo,
                            'avg FG attempts opp':FGAo,
                            'Avg 3P made opp':threePo,
                            'avg 3P attempts opp':threePAo,
                            'avg FT made opp':FTo,
                            'avg FT attempts opp':FTAo,
                            'avg Off Reb opp':ORBo,
                            'Avg Team Reb opp':TRBo,
                            'Avg Assists opp':ASTo,
                            'Avg Steals opp':STLo,
                            'Avg Blocks opp':BLKo,
                            'Avg turnovers opp':TOVo,
                            'Avg Pers Fouls opp':PFo})


finalDataF.to_csv('dataSEC.csv',sep=',')