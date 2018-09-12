# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:42:39 2018

@author: Mike
"""

import pandas as pd

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
qbs_sm = qbs.drop(['Lg','Result','Week','Cmp','Cmp%','Sk','Yds.1','AY/A'], axis = 1)
rbs_sm = rbs.drop(['Lg','Result','Week'], axis = 1)
wrs_sm = wrs.drop(['Lg','Result','Week','Ctch%','Y/Tgt'], axis = 1)

qbs_sm['Date'] = qbs_sm['Date'].str[2:4]
rbs_sm['Date'] = rbs_sm['Date'].str[2:4]
wrs_sm['Date'] = wrs_sm['Date'].str[2:4]

qbs_sm['Away'] = (qbs_sm['Away'] == '@')*1
rbs_sm['Away'] = (rbs_sm['Away'] == '@')*1
wrs_sm['Away'] = (wrs_sm['Away'] == '@')*1

qbs_sm = pd.get_dummies(qbs_sm, columns = ['Day'])
rbs_sm = pd.get_dummies(rbs_sm, columns = ['Day'])
wrs_sm = pd.get_dummies(wrs_sm, columns = ['Day'])

#Filling points/game from prev year
def prev_qb_yr_stats(row):
    try:
        value = qbs_sm[qbs_sm['Player'] == row['Player']].groupby('Date')['dk_pts'].get_group(str(int(row['Date'])-1)).mean()
    except KeyError:
        value = 0
    return value
def prev_rb_yr_stats(row):
    try:
        value = rbs_sm[rbs_sm['Player'] == row['Player']].groupby('Date')['dk_pts'].get_group(str(int(row['Date'])-1)).mean()
    except KeyError:
        value = 0
    return value
def prev_wr_yr_stats(row):
    try:
        value = wrs_sm[wrs_sm['Player'] == row['Player']].groupby('Date')['dk_pts'].get_group(str(int(row['Date'])-1)).mean()
    except KeyError:
        value = 0
    return value

qbs_sm['prev_yr_avg_pts'] = qbs_sm.apply(prev_qb_yr_stats,axis=1)
rbs_sm['prev_yr_avg_pts'] = rbs_sm.apply(prev_rb_yr_stats,axis=1)
wrs_sm['prev_yr_avg_pts'] = wrs_sm.apply(prev_wr_yr_stats,axis=1)


#Filling points in prev game
def prev_qb_gm_stats(row):
    try:
        value = qbs_sm[(qbs_sm['Player'] == row['Player']) & (qbs_sm['Date'] == row['Date']) & (qbs_sm['G#'] == row['G#']-1)]['dk_pts'].values[0]
    except IndexError:
        value = 0
    return value
def prev_rb_gm_stats(row):
    try:
        value = rbs_sm[(rbs_sm['Player'] == row['Player']) & (rbs_sm['Date'] == row['Date']) & (rbs_sm['G#'] == row['G#']-1)]['dk_pts'].values[0]
    except IndexError:
        value = 0
    return value
def prev_wr_gm_stats(row):
    try:
        value = wrs_sm[(wrs_sm['Player'] == row['Player']) & (wrs_sm['Date'] == row['Date']) & (wrs_sm['G#'] == row['G#']-1)]['dk_pts'].values[0]
    except IndexError:
        value = 0
    return value

qbs_sm['prev_gm_pts'] = qbs_sm.apply(prev_qb_gm_stats,axis=1)
rbs_sm['prev_gm_pts'] = rbs_sm.apply(prev_rb_gm_stats,axis=1)
wrs_sm['prev_gm_pts'] = wrs_sm.apply(prev_wr_gm_stats,axis=1)


#Opponents DEF rank







