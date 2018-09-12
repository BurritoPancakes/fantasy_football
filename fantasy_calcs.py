# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:42:39 2018

@author: Mike
"""

import pandas as pd
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

#conversation df
off_pts = pd.DataFrame(data = {'cat': ['pass_td','pass_yds','300_pass_yds','int','rush_td','rush_yds','100_rush_yds','rec_td','rec_yds','100_rec_yds','rec','ret_td','fum_lost','2_pt_conv','off_fum_rec_td'],'pts': [4,.04,3,-1,6,.1,3,6,.1,3,1,6,-1,2,6]})

#read in stats_dfs and fix column names
qbs = pd.read_csv('data/qb_stats_15_18.csv')
qbs.columns = ['Player', 'Age', 'Year', 'Lg', 'Tm', 'Away', 'Opp', 'Result', 'G#', 'Week', 'Day', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'Int', 'Rate', 'Sk', 'Yds.1', 'Y/A', 'AY/A']
rbs = pd.read_csv('data/rb_stats_15_18.csv')
rbs.columns = ['Player', 'Age', 'Year', 'Lg', 'Tm', 'Away', 'Opp', 'Result', 'G#', 'Week', 'Day', 'Att', 'Yds', 'Y/A', 'TD']
wrs = pd.read_csv('data/wr_stats_15_18.csv')
wrs.columns = ['Player', 'Age', 'Year', 'Lg', 'Tm', 'Away', 'Opp', 'Result', 'G#', 'Week', 'Day', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', 'Ctch%', 'Y/Tgt']               

#Calculating Fantasy Points
def qb_pts_dk(row):
    return (row['Yds']*off_pts['pts'][off_pts['cat'] == 'pass_yds'].values[0]) + (row['TD']*off_pts['pts'][off_pts['cat'] == 'pass_td'].values[0]) + ((row['Yds']>=300)*off_pts['pts'][off_pts['cat'] == '300_pass_yds'].values[0]) + (row['Int']*off_pts['pts'][off_pts['cat'] == 'int'].values[0])

def rb_pts_dk(row):
    return (row['Yds']*off_pts['pts'][off_pts['cat'] == 'rush_yds'].values[0]) + (row['TD']*off_pts['pts'][off_pts['cat'] == 'rush_td'].values[0]) + ((row['Yds']>=100)*off_pts['pts'][off_pts['cat'] == '100_rush_yds'].values[0])

def wr_pts_dk(row):
    return (row['Yds']*off_pts['pts'][off_pts['cat'] == 'rec_yds'].values[0]) + (row['TD']*off_pts['pts'][off_pts['cat'] == 'rec_td'].values[0]) + ((row['Yds']>=100)*off_pts['pts'][off_pts['cat'] == '100_rec_yds'].values[0]) + (row['Rec']*off_pts['pts'][off_pts['cat'] == 'rec'].values[0])


qbs['dk_pts'] = qbs.apply(qb_pts_dk,axis = 1)
rbs['dk_pts'] = rbs.apply(rb_pts_dk,axis = 1)
wrs['dk_pts'] = wrs.apply(wr_pts_dk,axis = 1)


#Fixing and deleting a few other features
qbs_sm = qbs.drop(['Lg','Result','Cmp','Cmp%','Sk','Yds.1','AY/A'], axis = 1)
rbs_sm = rbs.drop(['Lg','Result'], axis = 1)
wrs_sm = wrs.drop(['Lg','Result','Ctch%','Y/Tgt'], axis = 1)

qbs_sm['Year'] = qbs_sm['Year'].str[2:4]
rbs_sm['Year'] = rbs_sm['Year'].str[2:4]
wrs_sm['Year'] = wrs_sm['Year'].str[2:4]

qbs_sm['Away'] = (qbs_sm['Away'] == '@')*1
rbs_sm['Away'] = (rbs_sm['Away'] == '@')*1
wrs_sm['Away'] = (wrs_sm['Away'] == '@')*1

qbs_sm = pd.get_dummies(qbs_sm, columns = ['Day'])
rbs_sm = pd.get_dummies(rbs_sm, columns = ['Day'])
wrs_sm = pd.get_dummies(wrs_sm, columns = ['Day'])

#Filling points/game from prev year
def prev_qb_yr_stats(row):
    try:
        value = qbs_sm[qbs_sm['Player'] == row['Player']].groupby('Year')['dk_pts'].get_group(str(int(row['Year'])-1)).mean()
    except KeyError:
        value = 0
    return value
def prev_rb_yr_stats(row):
    try:
        value = rbs_sm[rbs_sm['Player'] == row['Player']].groupby('Year')['dk_pts'].get_group(str(int(row['Year'])-1)).mean()
    except KeyError:
        value = 0
    return value
def prev_wr_yr_stats(row):
    try:
        value = wrs_sm[wrs_sm['Player'] == row['Player']].groupby('Year')['dk_pts'].get_group(str(int(row['Year'])-1)).mean()
    except KeyError:
        value = 0
    return value

qbs_sm['prev_yr_avg_pts'] = qbs_sm.apply(prev_qb_yr_stats,axis=1)
rbs_sm['prev_yr_avg_pts'] = rbs_sm.apply(prev_rb_yr_stats,axis=1)
wrs_sm['prev_yr_avg_pts'] = wrs_sm.apply(prev_wr_yr_stats,axis=1)


#Filling points in prev game
def prev_qb_gm_stats(row):
    try:
        value = qbs_sm[(qbs_sm['Player'] == row['Player']) & (qbs_sm['Year'] == row['Year']) & (qbs_sm['Week'] == row['Week']-1)]['dk_pts'].values[0]
    except IndexError:
        value = 0
    return value
def prev_rb_gm_stats(row):
    try:
        value = rbs_sm[(rbs_sm['Player'] == row['Player']) & (rbs_sm['Year'] == row['Year']) & (rbs_sm['Week'] == row['Week']-1)]['dk_pts'].values[0]
    except IndexError:
        value = 0
    return value
def prev_wr_gm_stats(row):
    try:
        value = wrs_sm[(wrs_sm['Player'] == row['Player']) & (wrs_sm['Year'] == row['Year']) & (wrs_sm['Week'] == row['Week']-1)]['dk_pts'].values[0]
    except IndexError:
        value = 0
    return value

qbs_sm['prev_gm_pts'] = qbs_sm.apply(prev_qb_gm_stats,axis=1)
rbs_sm['prev_gm_pts'] = rbs_sm.apply(prev_rb_gm_stats,axis=1)
wrs_sm['prev_gm_pts'] = wrs_sm.apply(prev_wr_gm_stats,axis=1)


#Opponents DEF rank. Not sure where to get data for this.

#Fixing team names for consistency (transforming to most recent names)
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

qbs_sm['New_Tm'] = qbs_sm.apply(tm_name_change, axis=1)
qbs_sm['New_Opp'] = qbs_sm.apply(opp_name_change, axis=1)
rbs_sm['New_Tm'] = rbs_sm.apply(tm_name_change, axis=1)
rbs_sm['New_Opp'] = rbs_sm.apply(opp_name_change, axis=1)
wrs_sm['New_Tm'] = wrs_sm.apply(tm_name_change, axis=1)
wrs_sm['New_Opp'] = wrs_sm.apply(opp_name_change, axis=1)

#One-hot team and opp (might be able to give us indications of which teams are the best at offense and defense?)
qbs_sm = pd.get_dummies(qbs_sm, columns = ['New_Tm','New_Opp'])
rbs_sm = pd.get_dummies(rbs_sm, columns = ['New_Tm','New_Opp'])
wrs_sm = pd.get_dummies(wrs_sm, columns = ['New_Tm','New_Opp'])

#Add variable to show how many fantasy points the defense let up the previous week
def def_prev_gm_pts(row):
    try:
        value = rbs_sm[rbs_sm['New_Opp'] == row['New_Opp']].groupby(['Week','Year']).sum()['dk_pts'][row['Week']-1][row['Year']]
    except KeyError:
        try:
            value = rbs_sm[rbs_sm['New_Opp'] == row['New_Opp']].groupby(['Week','Year']).sum()['dk_pts'][row['Week']-2][row['Year']]
        except IndexError:
            value = rbs_sm[rbs_sm['New_Opp'] == row['New_Opp']].groupby(['Week','Year']).sum()['dk_pts'][16][str(int(row['Year'])-1)]
    except IndexError:
        try:
            value = rbs_sm[rbs_sm['New_Opp'] == row['New_Opp']].groupby(['Week','Year']).sum()['dk_pts'][16][str(int(row['Year'])-1)]
        except KeyError:
            value = 0
    return value 

rbs_sm['def_prev_gm_pts'] = rbs_sm.apply(def_prev_gm_pts,axis=1)

#rbs_sm[['Player','Week','Year','New_Tm','New_Opp','Yds','TD','def_prev_gm_pts']][(rbs_sm['Year'] == '15') & (rbs_sm['New_Opp'] == 'LAR')]

#First take on modeling

qbs_mdl = qbs_sm.drop(['Year','New_Tm','New_Opp','Att','Yds','Y/A','TD','Int','Rate'], axis = 1)
rbs_mdl = rbs_sm.drop(['Year','New_Tm','New_Opp','Att','Yds','Y/A','TD'], axis = 1)
wrs_mdl = wrs_sm.drop(['Year','New_Tm','New_Opp','Tgt','Rec','Yds','Y/R','TD'], axis = 1)


#Starting with just rbs_mdl

#label encoding players

rbs_mdl_trim = rbs_mdl.select_dtypes(exclude=['object'])
players = rbs_mdl.select_dtypes(include=['object'])
d = defaultdict(LabelEncoder)
fit = players.apply(lambda x: d[x.name].fit_transform(x))
model_df = players.apply(lambda x: d[x.name].transform(x))
model_df = model_df[players.columns]
rbs_mdl_trim = pd.concat([rbs_mdl_trim.reset_index(drop=True), model_df], axis=1)

#Modeling begins here
y = rbs_mdl_trim['dk_pts']
X = rbs_mdl_trim.drop(['dk_pts'], axis = 1)

# Split the data into train, test, validation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

##light gb
#model = XGBClassifier()
#model.fit(X_train, y_train)
#
## make predictions for test data
#yprob = model.predict(X_test)
#ypred = yprob >= .5
#
##View Results
#print(classification_report(y_test, ypred))
#
#cm = confusion_matrix(y_test, ypred)
#tn, fp, fn, tp = cm.ravel()
#
#print("True Negative: ", tn)
#print("False Positive: ", fp)
#print("False Negative: ", fn)
#print("True Positive: ", tp)


#LinearRegression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

X_train.columns
X_train_sm = X_train[['Age','prev_yr_avg_pts','prev_gm_pts','def_prev_gm_pts']]
X_test_sm = X_test[['Age','prev_yr_avg_pts','prev_gm_pts','def_prev_gm_pts']]


lm.fit(X_train_sm,y_train)

print(lm.intercept_)
print(lm.coef_)

results = pd.DataFrame({'features': X_train_sm.columns, 'estCoefs': lm.coef_})


ypred = lm.predict(X_test_sm)

pred_results = pd.DataFrame({'predictions': ypred,'actual': y_test})


join_back = rbs_sm.join(pred_results)
join_back = join_back.dropna()
#
week = 5
year = '17'
#
join_back.sort_values(['predictions'], ascending = False)[(join_back['Year'] == year) & (join_back['G#'] == week)][['Player','G#','Year','New_Tm','New_Opp','Att','Yds','TD','def_prev_gm_yds','prev_gm_pts','actual','predictions']].head(10)
join_back.sort_values(['actual'], ascending = False)[(join_back['Year'] == year) & (join_back['G#'] == week)][['Player','G#','Year','New_Tm','New_Opp','Att','Yds','TD','def_prev_gm_yds','prev_gm_pts','actual','predictions']].head(10)

