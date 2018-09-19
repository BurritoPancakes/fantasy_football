from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

#from selenium.webdriver.support.ui import WebDriverWait


class PFRScraper(object):
    """
    initiates a web scraper for Pro-Football-Reference Website
    """
    def __init__(self, driver_path='../webdrivers/chromedriver'):
        self.driver_path = driver_path
        self.year_range = [2015,2016,2017,2018]
        self.week_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    def scrape_qbs(self):
        """
        scrapes data from pfr's website for qb's, rb's, wr's, te's, def's
        """

        driver = webdriver.Chrome(executable_path=self.driver_path)

        #QB STATS:
        # c1val = 5 (min number of passes to get stats for)
        
        min_pass_att = 5
        qb_stats = pd.DataFrame()

        for year in self.year_range:
            for week in self.week_num:
                driver.get(f"https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game&year_min={year}&year_max={year}&season_start=1&season_end=-1&age_min=0&age_max=99&game_type=A&league_id=&team_id=&opp_id=&game_num_min=0&game_num_max=99&week_num_min={week}&week_num_max={week}&game_day_of_week=&game_location=&game_result=&handedness=&is_active=&is_hof=&c1stat=pass_att&c1comp=gt&c1val={min_pass_att}&c2stat=&c2comp=gt&c2val=&c3stat=&c3comp=gt&c3val=&c4stat=&c4comp=gt&c4val=&order_by=pass_rating&from_link=1")
                soup = BeautifulSoup(driver.page_source,"html.parser")
                stats_table = soup.find('table', id = 'results') 
                header = stats_table.find('thead')
                
                #First two are actually above the header and the third is rank which we don't grab when we get the data
                headers = [head.text for head in header.find_all('th')][3:]
                rows = []
                for row in stats_table.find_all('tr'):
                    rows.append([val.text for val in row.find_all('td')])
                    
                #Grabs the two header rows first
                rows = rows[2:]
                qb_stats_temp = pd.DataFrame(data = rows, columns = headers)
                qb_stats = qb_stats.append(qb_stats_temp, ignore_index = True)
                    
        driver.close()
        
        qb_stats_final = qb_stats.dropna()
        qb_stats_final.to_csv(f'data/qb_stats_{str(self.year_range[0])[2:]}_{str(self.year_range[-1])[2:]}.csv', index = False)

        return
    
    def scrape_wrs(self):
        """
        scrapes data from pfr's website for qb's, rb's, wr's, te's, def's
        """

        driver = webdriver.Chrome(executable_path=self.driver_path)

        #WR STATS:

        wr_stats = pd.DataFrame()

        for year in self.year_range:
            for week in self.week_num:
                driver.get(f"https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game&year_min={year}&year_max={year}&season_start=1&season_end=-1&age_min=0&age_max=99&game_type=A&league_id=&team_id=&opp_id=&game_num_min=0&game_num_max=99&week_num_min={week}&week_num_max={week}&game_day_of_week=&game_location=&game_result=&handedness=&is_active=&is_hof=&c1stat=rec&c1comp=gt&c1val=1&c2stat=&c2comp=gt&c2val=&c3stat=&c3comp=gt&c3val=&c4stat=&c4comp=gt&c4val=&order_by=rec_yds&from_link=1")
                soup = BeautifulSoup(driver.page_source,"html.parser")
                stats_table = soup.find('table', id = 'results') 
                header = stats_table.find('thead')
                
                #First two are actually above the header and the third is rank which we don't grab when we get the data
                headers = [head.text for head in header.find_all('th')][3:]
                rows = []
                for row in stats_table.find_all('tr'):
                    rows.append([val.text for val in row.find_all('td')])
                    
                #Grabs the two header rows first
                rows = rows[2:]
                wr_stats_temp = pd.DataFrame(data = rows, columns = headers)
                wr_stats = wr_stats.append(wr_stats_temp, ignore_index = True)
                    
        driver.close()

        wr_stats_final = wr_stats.dropna()
        wr_stats_final.to_csv(f'data/wr_stats_{str(self.year_range[0])[2:]}_{str(self.year_range[-1])[2:]}.csv', index = False)
        
        return
    
    def scrape_rbs(self):
        """
        scrapes data from pfr's website for qb's, rb's, wr's, te's, def's
        """

        driver = webdriver.Chrome(executable_path=self.driver_path)

        #RB STATS:

        rb_stats = pd.DataFrame()

        for year in self.year_range:
            for week in self.week_num:
                driver.get(f"https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game&year_min={year}&year_max={year}&season_start=1&season_end=-1&age_min=0&age_max=99&game_type=A&league_id=&team_id=&opp_id=&game_num_min=0&game_num_max=99&week_num_min={week}&week_num_max={week}&game_day_of_week=&game_location=&game_result=&handedness=&is_active=&is_hof=&c1stat=rush_att&c1comp=gt&c1val=1&c2stat=&c2comp=gt&c2val=&c3stat=&c3comp=gt&c3val=&c4stat=&c4comp=gt&c4val=&order_by=rush_yds&from_link=1")
                soup = BeautifulSoup(driver.page_source,"html.parser")
                stats_table = soup.find('table', id = 'results') 
                header = stats_table.find('thead')
                
                #First two are actually above the header and the third is rank which we don't grab when we get the data
                headers = [head.text for head in header.find_all('th')][3:]
                rows = []
                for row in stats_table.find_all('tr'):
                    rows.append([val.text for val in row.find_all('td')])
                    
                #Grabs the two header rows first
                rows = rows[2:]
                rb_stats_temp = pd.DataFrame(data = rows, columns = headers)
                rb_stats = rb_stats.append(rb_stats_temp, ignore_index = True)
                    
        driver.close()

        rb_stats_final = rb_stats.dropna()
        rb_stats_final.to_csv(f'data/rb_stats_{str(self.year_range[0])[2:]}_{str(self.year_range[-1])[2:]}.csv', index = False)
        
        return 
    
    
    def get_players_and_positions(self):
        """
        scrapes position and team data from pfr's website for qb's, rb's, wr's, te's
        """

        driver = webdriver.Chrome(executable_path=self.driver_path)

        #All player info:

        players = pd.DataFrame()

        for year in self.year_range:
            driver.get(f"https://www.pro-football-reference.com/years/{year}/fantasy.htm")
            soup = BeautifulSoup(driver.page_source,"html.parser")
            stats_table = soup.find('table', id = 'fantasy') 
            header = stats_table.find('thead')
            headers = [head.text for head in header.find_all('th')][11:]
            rows = []
            for row in stats_table.find_all('tr'):
                rows.append([val.text for val in row.find_all('td')])
            #Grabs the two header rows first
            rows = rows[2:]
            players_temp = pd.DataFrame(data = rows, columns = headers)
            players = players.append(players_temp, ignore_index = True)
            
            players_final = players.dropna()
            players_final.columns = ['Player', 'Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'TD', 'Int','Att', 'Yds', 'Y/A', 'TD', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', '2PM','2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']
            players_final.to_csv(f'data/player_info_{str(year)}.csv', index = False) 
                    
        driver.close() 
        
        return 

    
test = PFRScraper()

#test.scrape_wrs()
#test.scrape_rbs()
#test.scrape_qbs()
test.get_players_and_positions()

