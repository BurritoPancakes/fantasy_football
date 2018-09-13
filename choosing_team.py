#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:52:15 2018

@author: mmcshan1
"""

#We need to:
    #1. Take in the dataframes with point predictions
    #1A. Combine dataframes to get total point preds for each player (qb rush yds, rb rec yds, etc.)
    #2. Determine cost of each player
    #3. Determine the players that we absolutely want or do not want
    #4. Logic to understand positions
    #5. Run an optimization algorithm to fit under the price

import pandas as pd
    
#set parameters here
pred_path = 'predictions/'
resource_path = 'resources/'
curr_week = 2
curr_year = '18'
    

#reading in prediction dataframes and combining them
#structures: | Player | Position | Week | Year | Team | Opponent | Prediction | 
qbs = pd.read_csv(pred_path + f'qbs_pred_{curr_year}_{curr_week}.csv')
rbs = pd.read_csv(pred_path + f'rbs_pred_{curr_year}_{curr_week}.csv')
wrs = pd.read_csv(pred_path + f'wrs_pred_{curr_year}_{curr_week}.csv')

tot_pts_pred = qbs.append(rbs).append(wrs)



#resource dataframes
player_costs = pd.read_csv(resource_path + f'player_costs_{curr_year}_{curr_week}.csv')
player_positions = pd.read_csv(resource_path + f'player_positions_{curr_year}_{curr_week}.csv')

want_players = ['just a list of names of players','Carson Wentz','Zach Ertz','etc.']
dont_want_players = ['just a list of names of players','Tom Brady','Eli Manning','etc.']

#