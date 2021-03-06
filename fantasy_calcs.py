# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:42:39 2018

@author: Mike
"""

#In this script we are writing the code for:
    #Calculating fantasy points per player
    #Fixing team names who have changed
    #Points/game from prev year
    #Prev game points
    #Points opp def let up (by position) prev week
    #etc.


#New way of creating dataframes:
    #Get rid of players who don't play. Only make model for top X players at each position (but still get their rushing, receiving, and passing stats)


#Imports
import re
import pandas as pd
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

#Functions:
    #tm_name_change()
    #opp_name_change()
    #qb_pts_dk()
    #prev_qb_yr_dk_pts()

#Changing team names if they moved cities
team_name_change = pd.DataFrame({'old_name' : ['SDG','STL'],'new_name':['LAC','LAR']})

def tm_name_change(row):
    if(row['Tm'] in list(team_name_change['old_name'])):
        val = team_name_change['new_name'][team_name_change['old_name'] == row['Tm']].values[0]
    else:
        val = row['Tm']
    return val
def opp_name_change(row):
    if(row['Opp'] in list(team_name_change['old_name'])):
        val = team_name_change['new_name'][team_name_change['old_name'] == row['Opp']].values[0]
    else:
        val = row['Opp']
    return val

#Calculating how many fantasy points each player got every game
def qb_pts_dk(row):
    return (row['Yds']*off_pts['pts'][off_pts['cat'] == 'pass_yds'].values[0]) + (row['TD']*off_pts['pts'][off_pts['cat'] == 'pass_td'].values[0]) + ((row['Yds']>=300)*off_pts['pts'][off_pts['cat'] == '300_pass_yds'].values[0]) + (row['Int']*off_pts['pts'][off_pts['cat'] == 'int'].values[0])

def rb_pts_dk(row):
    return (row['Yds']*off_pts['pts'][off_pts['cat'] == 'rush_yds'].values[0]) + (row['TD']*off_pts['pts'][off_pts['cat'] == 'rush_td'].values[0]) + ((row['Yds']>=100)*off_pts['pts'][off_pts['cat'] == '100_rush_yds'].values[0])

def wr_pts_dk(row):
    return (row['Yds']*off_pts['pts'][off_pts['cat'] == 'rec_yds'].values[0]) + (row['TD']*off_pts['pts'][off_pts['cat'] == 'rec_td'].values[0]) + ((row['Yds']>=100)*off_pts['pts'][off_pts['cat'] == '100_rec_yds'].values[0]) + (row['Rec']*off_pts['pts'][off_pts['cat'] == 'rec'].values[0])

#Calculating avg. points/game for each player for the previous year
def prev_qb_yr_dk_pts(row):
    try:
        value = qbs_sm[qbs_sm['Player'] == row['Player']].groupby('Year')['dk_pts'].get_group(str(int(row['Year'])-1)).mean()
    except KeyError:
        value = 0
    return value
def prev_rb_yr_dk_pts(row):
    try:
        value = rbs_sm[rbs_sm['Player'] == row['Player']].groupby('Year')['dk_pts'].get_group(str(int(row['Year'])-1)).mean()
    except KeyError:
        value = 0
    return value
def prev_wr_yr_dk_pts(row):
    try:
        value = wrs_sm[wrs_sm['Player'] == row['Player']].groupby('Year')['dk_pts'].get_group(str(int(row['Year'])-1)).mean()
    except KeyError:
        value = 0
    return value

#Calculating points for each player for the previous three games
def prev_qb_gm_dk_pts(row):
    try:
        week_one = qbs_sm[(qbs_sm['Player'] == row['Player']) & (qbs_sm['Year'] == row['Year']) & (qbs_sm['Week'] == row['Week']-1)]['dk_pts'].values[0]
    except IndexError:
        week_one = 0
    try:
        week_two = qbs_sm[(qbs_sm['Player'] == row['Player']) & (qbs_sm['Year'] == row['Year']) & (qbs_sm['Week'] == row['Week']-2)]['dk_pts'].values[0]
    except IndexError:
        week_two = 0
    try:
        week_three = qbs_sm[(qbs_sm['Player'] == row['Player']) & (qbs_sm['Year'] == row['Year']) & (qbs_sm['Week'] == row['Week']-3)]['dk_pts'].values[0]
    except IndexError:
        week_three = 0
    return (week_one + week_two + week_three)/3
def prev_rb_gm_dk_pts(row):
    try:
        week_one = rbs_sm[(rbs_sm['Player'] == row['Player']) & (rbs_sm['Year'] == row['Year']) & (rbs_sm['Week'] == row['Week']-1)]['dk_pts'].values[0]
    except IndexError:
        week_one = 0
    try:
        week_two = rbs_sm[(rbs_sm['Player'] == row['Player']) & (rbs_sm['Year'] == row['Year']) & (rbs_sm['Week'] == row['Week']-2)]['dk_pts'].values[0]
    except IndexError:
        week_two = 0
    try:
        week_three = rbs_sm[(rbs_sm['Player'] == row['Player']) & (rbs_sm['Year'] == row['Year']) & (rbs_sm['Week'] == row['Week']-3)]['dk_pts'].values[0]
    except IndexError:
        week_three = 0
    return (week_one + week_two + week_three)/3
def prev_wr_gm_dk_pts(row):
    try:
        week_one = wrs_sm[(wrs_sm['Player'] == row['Player']) & (wrs_sm['Year'] == row['Year']) & (wrs_sm['Week'] == row['Week']-1)]['dk_pts'].values[0]
    except IndexError:
        week_one = 0
    try:
        week_two = wrs_sm[(wrs_sm['Player'] == row['Player']) & (wrs_sm['Year'] == row['Year']) & (wrs_sm['Week'] == row['Week']-2)]['dk_pts'].values[0]
    except IndexError:
        week_two = 0
    try:
        week_three = wrs_sm[(wrs_sm['Player'] == row['Player']) & (wrs_sm['Year'] == row['Year']) & (wrs_sm['Week'] == row['Week']-3)]['dk_pts'].values[0]
    except IndexError:
        week_three = 0
    return (week_one + week_two + week_three)/3


#Figuring out who goes in the model
players = pd.DataFrame()
for year in ['2015','2016','2017','2018']:
    players_yr = pd.read_csv(f'data/player_info_{year}.csv')
    players = players.append(players_yr)

players = players[players['FantPt'] >= 30]
players = players['Player'].unique()
players = [re.sub("[^ \w.]"," ",y).strip() for y in players]

#conversion df
off_pts = pd.DataFrame(data = {'cat': ['pass_td','pass_yds','300_pass_yds','int','rush_td','rush_yds','100_rush_yds','rec_td','rec_yds','100_rec_yds','rec','ret_td','fum_lost','2_pt_conv','off_fum_rec_td'],'pts': [4,.04,3,-1,6,.1,3,6,.1,3,1,6,-1,2,6]})

#read in stats_dfs and fix column names
qbs = pd.read_csv('data/qb_stats_15_18.csv')
qbs.columns = ['Player', 'Age', 'Year', 'Lg', 'Tm', 'Away', 'Opp', 'Result', 'G#', 'Week', 'Day', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'Int', 'Rate', 'Sk', 'Yds.1', 'Y/A', 'AY/A']
qbs = qbs[qbs['Player'].isin(players)]
rbs = pd.read_csv('data/rb_stats_15_18.csv')
rbs.columns = ['Player', 'Age', 'Year', 'Lg', 'Tm', 'Away', 'Opp', 'Result', 'G#', 'Week', 'Day', 'Att', 'Yds', 'Y/A', 'TD']
rbs = rbs[rbs['Player'].isin(players)]
wrs = pd.read_csv('data/wr_stats_15_18.csv')
wrs.columns = ['Player', 'Age', 'Year', 'Lg', 'Tm', 'Away', 'Opp', 'Result', 'G#', 'Week', 'Day', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', 'Ctch%', 'Y/Tgt']               
wrs = wrs[wrs['Player'].isin(players)]


#Calculating how many fantasy points a player got
qbs['dk_pts'] = qbs.apply(qb_pts_dk,axis = 1)
rbs['dk_pts'] = rbs.apply(rb_pts_dk,axis = 1)
wrs['dk_pts'] = wrs.apply(wr_pts_dk,axis = 1)


#Making dataframe a little smaller with features we will not use
qbs_sm = qbs.drop(['Lg','Result','Cmp','Cmp%','Sk','Yds.1','AY/A'], axis = 1)
rbs_sm = rbs.drop(['Lg','Result'], axis = 1)
wrs_sm = wrs.drop(['Lg','Result','Ctch%','Y/Tgt'], axis = 1)

#Fixing team names for consistency (transforming to most recent names)
qbs_sm['Tm'] = qbs_sm.apply(tm_name_change, axis=1)
qbs_sm['Opp'] = qbs_sm.apply(opp_name_change, axis=1)
rbs_sm['Tm'] = rbs_sm.apply(tm_name_change, axis=1)
rbs_sm['Opp'] = rbs_sm.apply(opp_name_change, axis=1)
wrs_sm['Tm'] = wrs_sm.apply(tm_name_change, axis=1)
wrs_sm['Opp'] = wrs_sm.apply(opp_name_change, axis=1)

#Fixing and deleting a few other features (Year, Away, Day)
qbs_sm['Year'] = qbs_sm['Year'].str[2:4]
rbs_sm['Year'] = rbs_sm['Year'].str[2:4]
wrs_sm['Year'] = wrs_sm['Year'].str[2:4]

qbs_sm['Away'] = (qbs_sm['Away'] == '@')*1
rbs_sm['Away'] = (rbs_sm['Away'] == '@')*1
wrs_sm['Away'] = (wrs_sm['Away'] == '@')*1

qbs_sm = pd.get_dummies(qbs_sm, columns = ['Day'])
rbs_sm = pd.get_dummies(rbs_sm, columns = ['Day'])
wrs_sm = pd.get_dummies(wrs_sm, columns = ['Day'])

#Creating DEF rankings by week dataframe
qb_def = qbs_sm.groupby(['Opp','Year','Week']).sum()[['dk_pts']].reset_index()

index_list = list()
for year in qb_def['Year'].unique():
    for week in qb_def['Week'].unique():
        index_list.append((year + '_' + week.astype(str)))
index_list.sort()

full_qb_def = pd.DataFrame(columns = qbs_sm['Opp'].unique(), index = index_list)
for item in full_qb_def.index:
    for team in full_qb_def.columns:
        try:
            year = item.split('_')[0]
            week = int(item.split('_')[1])
            temp_df = qb_def[(qb_def['Year']==year)&(qb_def['Week']==week)].sort_values(['Year','Week','dk_pts']).reset_index().reset_index()
            full_qb_def.loc[item,team] = temp_df['level_0'][temp_df['Opp'] == team].values[0]
        except IndexError:
            full_qb_def.loc[item,team] = 999

rb_def = rbs_sm.groupby(['Opp','Year','Week']).sum()[['dk_pts']].reset_index()

index_list = list()
for year in rb_def['Year'].unique():
    for week in rb_def['Week'].unique():
        index_list.append((year + '_' + week.astype(str)))
index_list.sort()

full_rb_def = pd.DataFrame(columns = rbs_sm['Opp'].unique(), index = index_list)
for item in full_rb_def.index:
    for team in full_rb_def.columns:
        try:
            year = item.split('_')[0]
            week = int(item.split('_')[1])
            temp_df = rb_def[(rb_def['Year']==year)&(rb_def['Week']==week)].sort_values(['Year','Week','dk_pts']).reset_index().reset_index()
            full_rb_def.loc[item,team] = temp_df['level_0'][temp_df['Opp'] == team].values[0]
        except IndexError:
            full_rb_def.loc[item,team] = 999

#Weird error with this value so I am setting it here. Will look into later.  
full_rb_def.loc['16_14','BUF'] = 31        
            
wr_def = wrs_sm.groupby(['Opp','Year','Week']).sum()[['dk_pts']].reset_index()

index_list = list()
for year in wr_def['Year'].unique():
    for week in wr_def['Week'].unique():
        index_list.append((year + '_' + week.astype(str)))
index_list.sort()

full_wr_def = pd.DataFrame(columns = wrs_sm['Opp'].unique(), index = index_list)
for item in full_wr_def.index:
    for team in full_wr_def.columns:
        try:
            year = item.split('_')[0]
            week = int(item.split('_')[1])
            temp_df = wr_def[(wr_def['Year']==year)&(wr_def['Week']==week)].sort_values(['Year','Week','dk_pts']).reset_index().reset_index()
            full_wr_def.loc[item,team] = temp_df['level_0'][temp_df['Opp'] == team].values[0]
        except IndexError:
            full_wr_def.loc[item,team] = 999
     
        
#Adding DEF prev 3 week avg. ranking
def qb_def_prev_gm_rank(row):
    curr_week = row['Week']
    curr_year = row['Year']
    opp = row['Opp']
    avg_rank = []
    for value in [1,2,3]:
        try:
            if(full_qb_def.loc[curr_year + '_' + str(curr_week - value),opp] == 999):
                rank = full_qb_def.loc[curr_year + '_' + str(curr_week - (value-1)),opp]
            else:
                rank = full_qb_def.loc[curr_year + '_' + str(curr_week - value),opp]
        except KeyError:
            if(curr_year == '15'):
                rank = 0
            else:
                rank = full_qb_def.loc[str(int(curr_year)-1) + '_' + str(17-value),opp]
        avg_rank.append(rank)
    return sum(avg_rank)/len(avg_rank)

qbs_sm['def_prev_gm_qb_rank'] = qbs_sm.apply(qb_def_prev_gm_rank,axis=1)

def rb_def_prev_gm_rank(row):
    curr_week = row['Week']
    curr_year = row['Year']
    opp = row['Opp']
    avg_rank = []
    for value in [1,2,3]:
        try:
            if(full_rb_def.loc[curr_year + '_' + str(curr_week - value),opp] == 999):
                rank = full_rb_def.loc[curr_year + '_' + str(curr_week - (value-1)),opp]
            else:
                rank = full_rb_def.loc[curr_year + '_' + str(curr_week - value),opp]
        except KeyError:
            if(curr_year == '15'):
                rank = 0
            else:
                rank = full_rb_def.loc[str(int(curr_year)-1) + '_' + str(17-value),opp]
        avg_rank.append(rank)
    return sum(avg_rank)/len(avg_rank)

rbs_sm['def_prev_gm_rb_rank'] = rbs_sm.apply(rb_def_prev_gm_rank,axis=1)

def wr_def_prev_gm_rank(row):
    curr_week = row['Week']
    curr_year = row['Year']
    opp = row['Opp']
    avg_rank = []
    for value in [1,2,3]:
        try:
            if(full_wr_def.loc[curr_year + '_' + str(curr_week - value),opp] == 999):
                rank = full_wr_def.loc[curr_year + '_' + str(curr_week - (value-1)),opp]
            else:
                rank = full_wr_def.loc[curr_year + '_' + str(curr_week - value),opp]
        except KeyError:
            if(curr_year == '15'):
                rank = 0
            else:
                rank = full_wr_def.loc[str(int(curr_year)-1) + '_' + str(17-value),opp]
        avg_rank.append(rank)
    return sum(avg_rank)/len(avg_rank)

wrs_sm['def_prev_gm_wr_rank'] = wrs_sm.apply(wr_def_prev_gm_rank,axis=1)

#Filling points/game from prev year
qbs_sm['prev_yr_avg_dk_pts'] = qbs_sm.apply(prev_qb_yr_dk_pts,axis=1)
rbs_sm['prev_yr_avg_dk_pts'] = rbs_sm.apply(prev_rb_yr_dk_pts,axis=1)
wrs_sm['prev_yr_avg_dk_pts'] = wrs_sm.apply(prev_wr_yr_dk_pts,axis=1)


#Filling points in prev 3 games
qbs_sm['prev_three_gm_pts'] = qbs_sm.apply(prev_qb_gm_dk_pts,axis=1)
rbs_sm['prev_three_gm_pts'] = rbs_sm.apply(prev_rb_gm_dk_pts,axis=1)
wrs_sm['prev_three_gm_pts'] = wrs_sm.apply(prev_wr_gm_dk_pts,axis=1)

#Adding in Vegas Odds Data
#HARD CODED DATES HERE. WHOOPS
vegas_odds = pd.read_csv('data/vegas_odds_15_18.csv')
vegas_odds = vegas_odds.drop_duplicates(['Date','Favorite','Spread','Underdog','Away Money Line','Home Money Line'])
#vegas_odds = vegas_odds[vegas_odds['Total'] != 'OFF']
del vegas_odds['Date']


def vegas_matchups(row):
    if(row['Favorite'].startswith('at ')):
        home = 1
    else:
        home = 0   
    return home

vegas_odds['Home'] = vegas_odds.apply(vegas_matchups, axis=1)
vegas_odds['Favorite'] = vegas_odds['Favorite'].map(lambda x: x.lstrip('at '))
vegas_odds['Underdog'] = vegas_odds['Underdog'].map(lambda x: x.lstrip('at '))

full_tm_names = ['Jets','Titans','Giants','Packers','Bengals','Broncos','Saints',
 'Colts','Bills','Raiders','49ers','Chargers','Bears','Cowboys','Patriots',
 'Steelers','Browns','Jaguars','Texans','Redskins','Panthers','Ravens',
 'Falcons','Eagles','Vikings','Dolphins','Chiefs','Rams','Cardinals',
 'Seahawks','Lions','Buccaneers']
abb_tm_name =['NYJ','TEN','NYG','GNB','CIN','DEN','NOR','IND','BUF','OAK','SFO','LAC','CHI','DAL','NWE','PIT','CLE','JAX','HOU','WAS','CAR',
              'BAL','ATL','PHI','MIN','MIA','KAN','LAR','ARI','SEA','DET','TAM']
team_conv_df = pd.DataFrame({'full_name':full_tm_names,'abb_name':abb_tm_name})

matchups = vegas_odds.merge(team_conv_df, left_on = 'Favorite',right_on = 'full_name')[['abb_name','Home','Underdog','Year','Week','Home Money Line','Away Money Line','Spread','Total']]
matchups.columns = ['Favorite','Home','Underdog','Year','Week','Home Money Line','Away Money Line','Spread','Total']
matchups = matchups.merge(team_conv_df, left_on = 'Underdog',right_on = 'full_name')[['Favorite','Home','abb_name','Year','Week','Home Money Line','Away Money Line','Spread','Total']]
matchups.columns = ['Favorite','Home','Underdog','Year','Week','Home Money Line','Away Money Line','Spread','Total']
matchups['Year'] = matchups['Year'].apply(str)
matchups['Year'] = matchups['Year'].str[2:]

#Combining Vegas data to model data

testx = rbs_sm.merge(matchups, left_on = ['Tm','Year','Week'],right_on = ['Favorite','Year','Week'])[['Player','Age','Year','Tm','Away','Opp','G#','Week','Att','Yds','def_prev_gm_rb_rank','prev_yr_avg_dk_pts','prev_three_gm_pts','Spread','Total','Home','dk_pts']]
testx['proj_team_pts'] = (testx['Total'] - testx['Spread'])/2
testy = rbs_sm.merge(matchups, left_on = ['Tm','Year','Week'],right_on = ['Underdog','Year','Week'])[['Player','Age','Year','Tm','Away','Opp','G#','Week','Att','Yds','def_prev_gm_rb_rank','prev_yr_avg_dk_pts','prev_three_gm_pts','Spread','Total','Home','dk_pts']]
testy['proj_team_pts'] = (testy['Total'] + testy['Spread'])/2

final_vegas = testx.append(testy, ignore_index = True)
del final_vegas['Home']
del final_vegas['Spread']


#Taking out 2015 since all our good variables are all zeroes
rbs_sm_mdl = final_vegas[final_vegas['Year'] != '15']
rbs_sm_mdl = rbs_sm_mdl[rbs_sm_mdl['Att'] > 3]


#First take on modeling
qbs_mdl = qbs_sm.drop(['Year','Tm','Opp','Att','Yds','Y/A','TD','Int','Rate'], axis = 1)
rbs_mdl = rbs_sm_mdl.drop(['Player','Tm','Opp','Year','Att','Yds'], axis = 1)
wrs_mdl = wrs_sm.drop(['Year','Tm','Opp','Tgt','Rec','Yds','Y/R','TD'], axis = 1)


#Starting with just rbs_mdl
    #Remove people who have very few attempts? (less than 3?)
    #Remove people


#label encoding players
#rbs_mdl_trim = rbs_mdl.select_dtypes(exclude=['object'])
#players = rbs_mdl.select_dtypes(include=['object'])
#d = defaultdict(LabelEncoder)
#fit = players.apply(lambda x: d[x.name].fit_transform(x))
#model_df = players.apply(lambda x: d[x.name].transform(x))
#model_df = model_df[players.columns]
#rbs_mdl_trim = pd.concat([rbs_mdl_trim.reset_index(drop=True), model_df], axis=1)

#Modeling begins here
y = rbs_mdl.loc[:,'dk_pts']
X = rbs_mdl.drop(['dk_pts'], axis = 1)

y[y <= 0] = .2
log_y = np.log(y)

# Split the data into train, test, validation 
X_train, X_test, y_train, y_test = train_test_split(X, log_y, test_size=.3)


#Try GLM or Poisson models instead of linear because the outcome variable is skewed
import matplotlib.pyplot as plt
plt.hist(y_train)
#LinearRegression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
#import statsmodels.api as sm
## Instantiate a gamma family model with the default link function.
#gamma_model = sm.GLM()
#gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
#gamma_results = gamma_model.fit()
#print(gamma_results.summary())

    
#ADD IN AVG # OF TD'S IN LAST X WEEKS
#ADD IN AVG # ATTEMPTS/TGTS IN PAST X WEEKS
#Figure out how to capture that X factor which separates people a little more need to predict outliers?
X_train_sm = X_train[['Age', 'Away','def_prev_gm_rb_rank', 'prev_yr_avg_dk_pts','prev_three_gm_pts','Total','proj_team_pts']]
X_test_sm = X_test[['Age', 'Away','def_prev_gm_rb_rank', 'prev_yr_avg_dk_pts','prev_three_gm_pts','Total','proj_team_pts']]


lm.fit(X_train_sm,y_train)

print(lm.intercept_)
print(lm.coef_)



results = pd.DataFrame({'features': X_train_sm.columns, 'estCoefs': lm.coef_})
ypred = lm.predict(X_test_sm)
pred_results = pd.DataFrame({'predictions': ypred,'actual': y_test})

join_back = rbs_sm_mdl.join(pred_results)
join_back = join_back.dropna()
#
week = 14
year = '16'
#
join_back.sort_values(['predictions'], ascending = False)[(join_back['Year'] == year) & (join_back['G#'] == week)][['Player','G#','Year','Tm','Opp','Att','Yds','TD','def_prev_gm_rb_rank','prev_yr_avg_dk_pts','prev_three_gm_pts','dk_pts','predictions']].head(10)
join_back.sort_values(['actual'], ascending = False)[(join_back['Year'] == year) & (join_back['G#'] == week)][['Player','G#','Year','Tm','Opp','Att','Yds','TD','def_prev_gm_rb_rank','prev_three_gm_pts','dk_pts','predictions']].head(10)



output_preds_rb = join_back[['Player','Year','G#','Opp','Tm','predictions']]




#Plotting variables against coefficients
import matplotlib.pyplot as plt
plt.scatter(rbs_sm_mdl['def_prev_gm_rb_rank'],rbs_sm_mdl['dk_pts'])


join_back.sort_values(['predictions'], ascending = False)[['Player','G#','Year','Tm','Opp','Att','Yds','TD','def_prev_gm_rb_rank','prev_yr_avg_dk_pts','prev_three_gm_pts','actual','predictions']].head(10)

