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
https://www.pro-football-reference.com/years/2018/week_3.htm


#Return table that looks like:
    # | Away Team | @ | Home Team | Day | | | | 
    
    
    
year = 2018
week = 3
    
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
    
all_matchups = all_matchups.reset_index(drop = True)
all_matchups['Away'] = '@'
    

players = pd.read_csv(f'data/player_info_{year}.csv') 

pred_df = pd.DataFrame(columns = ["Player","Tm","FantPos",""])


























