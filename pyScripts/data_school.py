import pandas as pd
import numpy as np

def split(word):
    return [char for char in word]

namesData = pd.read_csv('./names_school.csv',delimiter=',')

bigstat_team = []

common_names = namesData['Common Name'].tolist()
site_names = namesData['site name'].tolist()
conference = namesData['conference'].tolist()

stats = ['FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF']

oppStat= []
teamStat = []
outcome = []
points_school = []
points_opp = []
location = []
conf = []
opp_conf = []

years = years = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]

for year in years:
    print(year)
    done = []
    for team in site_names:

        done.append(common_names[site_names.index(team)])

        df = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{team}/{year}-gamelogs.html')[0]
        opponents = df['Unnamed: 3_level_0']['Opp'].tolist()
        threePA = df['Unnamed: 5_level_0']['Tm']
        oppPoints = df['Unnamed: 6_level_0']['Opp']
        place = df['Unnamed: 2_level_0']['Unnamed: 2_level_1']
        threePAschool = []
        for i in threePA:
            try:
                threePAschool.append(int(i))
            except ValueError:
                continue


        for i in range(0,len(opponents)):
            if opponents[i] not in done:
                if opponents[i] in common_names:
                    points_school.append(threePA[i])
                    points_opp.append(oppPoints[i])
                    if place[i] == 'NaN':
                        location.append('home')
                    else:
                        location.append(place[i])
                    conf.append(conference[site_names.index(team)])
                    opp_conf.append(conference[common_names.index(opponents[i])])
                    stat_array_team = []
                    stat_array_opp = []
                    for stat in stats:

                        temp_array = []
                        stat_array = df['School'][stat]
                        opp_stat_array = df['Opponent'][stat]
                        stat_array_team.append(stat_array[i])
                        stat_array_opp.append(opp_stat_array[i])
                    
                    teamStat.append(stat_array_team)
                    oppStat.append(stat_array_opp)

            else:
                continue


#'FG','FGA','3P','3PA','FT','FTA','ORB','TRB','AST','STL','BLK','TOV','PF'

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

print(teamStat)
print(oppStat)

for array in teamStat:
    try:
        FG.append(array[0])
        FGA.append(array[1])
        threeP.append(array[2])
        threePA.append(array[3])
        FT.append(array[4])
        FTA.append(array[5])
        ORB.append(array[6])
        TRB.append(array[7])
        AST.append(array[8])
        STL.append(array[9])
        BLK.append(array[10])
        TOV.append(array[11])
        PF.append(array[12])
    except IndexError:
        FG.append(0)
        FGA.append(0)
        threeP.append(0)
        threePA.append(0)
        FT.append(0)
        FTA.append(0)
        ORB.append(0)
        TRB.append(0)
        AST.append(0)
        STL.append(0)
        BLK.append(0)
        TOV.append(0)
        PF.append(0)

for array in oppStat:
    try:
        FGo.append(array[0])
        FGAo.append(array[1])
        threePo.append(array[2])
        threePAo.append(array[3])
        FTo.append(array[4])
        FTAo.append(array[5])
        ORBo.append(array[6])
        TRBo.append(array[7])
        ASTo.append(array[8])
        STLo.append(array[9])
        BLKo.append(array[10])
        TOVo.append(array[11])
        PFo.append(array[12])
    except IndexError:
        FGo.append(0)
        FGAo.append(0)
        threePo.append(0)
        threePAo.append(0)
        FTo.append(0)
        FTAo.append(0)
        ORBo.append(0)
        TRBo.append(0)
        ASTo.append(0)
        STLo.append(0)
        BLKo.append(0)
        TOVo.append(0)
        PFo.append(0)

print(len(conf))
print(len(location))
print(len(points_school))
print(len(FG))


finalDataF = pd.DataFrame({'Conference':conf,
                            'Location':location,
                            'Points':points_school,
                            'FG made':FG,
                            'FG attempts':FGA,
                            '3P made':threeP,
                            '3P attempts':threePA,
                            'FT made':FT,
                            'FT attempts':FTA,
                            'Off Reb':ORB,
                            'Team Reb':TRB,
                            'Assists':AST,
                            'Steals':STL,
                            'Blocks':BLK,
                            'turnovers':TOV,
                            'Pers Fouls':PF,
                            'Opponent Conf':opp_conf,
                            'Points opp':points_opp,
                            'FG made opp':FGo,
                            'FG attempts opp':FGAo,
                            '3P made opp':threePo,
                            '3P attempts opp':threePAo,
                            'FT made opp':FTo,
                            'FT attempts opp':FTAo,
                            'Off Reb opp':ORBo,
                            'Team Reb opp':TRBo,
                            'Assists opp':ASTo,
                            'Steals opp':STLo,
                            'Blocks opp':BLKo,
                            'turnovers opp':TOVo,
                            'Pers Fouls opp':PFo})


finalDataF.to_csv('dataSchoolnew.csv',sep=',')