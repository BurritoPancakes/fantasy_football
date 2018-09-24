# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:14:53 2018

@author: Mike
"""


from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

#STEPS:
    #Grab games from pfr
    #
    
    
    
    
    
    


#Use this link to scrape the upcoming games for next week (get team names and then grab all the players by team code)
    #Get date and teams
#https://www.pro-football-reference.com/years/2018/week_3.htm


#Return table that looks like:
    # | Away Team | @ | Home Team | Day | | | | 
    
    
    
year = 2018
week = 4
    
driver = webdriver.Chrome(executable_path='webdrivers/chromedriver')
driver.get(f"https://www.pro-football-reference.com/years/{year}/week_{week}.htm")
soup = BeautifulSoup(driver.page_source,"html.parser")

week_games = soup.findAll("div", {"class":"game_summary nohover"}) 

all_matchups = pd.DataFrame(columns = ["Day","Tm","Opp"])
for matchup in week_games:
    day = matchup.find("tr",{"class":"date"}).text
    team_one = matchup.find_all("a")[0].text
    team_two = matchup.find_all("a")[2].text
    matchup_df = pd.DataFrame({"Day": [day],"Tm":[team_one],"Opp":[team_two]})
    
    all_matchups = all_matchups.append(matchup_df)
    
driver.close()

all_matchups = all_matchups.reset_index(drop = True)
all_matchups['Away'] = '@'

full_tm_names = ['New York Jets','Tennessee Titans','New York Giants','Green Bay Packers','Cincinnati Bengals','Denver Broncos','New Orleans Saints',
 'Indianapolis Colts','Buffalo Bills','Oakland Raiders','San Francisco 49ers','Los Angeles Chargers','Chicago Bears','Dallas Cowboys','New England Patriots',
 'Pittsburgh Steelers','Cleveland Browns','Jacksonville Jaguars','Houston Texans','Washington Redskins','Carolina Panthers','Baltimore Ravens',
 'Atlanta Falcons','Philadelphia Eagles','Minnesota Vikings','Miami Dolphins','Kansas City Chiefs','Los Angeles Rams','Arizona Cardinals',
 'Seattle Seahawks','Detroit Lions','Tampa Bay Buccaneers']
abb_tm_name =['NYJ','TEN','NYG','GNB','CIN','DEN','NOR','IND','BUF','OAK','SFO','LAC','CHI','DAL','NWE','PIT','CLE','JAX','HOU','WAS','CAR',
              'BAL','ATL','PHI','MIN','MIA','KAN','LAR','ARI','SEA','DET','TAM']
team_conv_df = pd.DataFrame({'full_name':full_tm_names,'abb_name':abb_tm_name})


matchups = all_matchups.join(team_conv_df)[['Day','abb_name','Away','Opp']]
matchups = matchups.merge(team_conv_df, left_on = 'Opp',right_on = 'full_name')[['Day', 'abb_name_x', 'Away', 'abb_name_y']]
    

players = pd.read_csv(f'data/player_info_{year}.csv') 

pred_df = pd.DataFrame({"Player":players['Player'].unique()})

pred_df = pred_df.merge(players[['Player','Tm','FantPos','Age']], on = 'Player')
pred_df = pred_df.dropna()

away_teams = pred_df.merge(matchups, left_on = 'Tm',right_on='abb_name_x')[['Player','Age','Tm','FantPos','abb_name_y','Away']]
away_teams.columns = ['Player','Age','Tm','FantPos','Opp','Away']
home_teams = pred_df.merge(matchups, left_on = 'Tm',right_on='abb_name_y')[['Player','Age','Tm','FantPos','abb_name_x']]
home_teams.columns = ['Player','Age','Tm','FantPos','Opp']


df = away_teams.append(home_teams)[['Player','Age','FantPos','Tm','Away','Opp']].reset_index(drop=True)
df['Away'] = (df['Away'] == '@')*1

#Add in the other columns that go into the model ##'Week','Day_Thu','Day_Mon','def_prev_gm_rb_rank', 'prev_yr_avg_dk_pts','prev_three_gm_pts'


#Run rushing yds pred
#Run receiving yds pred
#Run throwing yds pred

















