import json
from api.local_server import CRAG

class SportsTool:
    def __init__(self):
        self.api = CRAG()
        self.nba_teams = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]
        self.nba_teams_alter = {
            "Atlanta Hawks": ["Hawks", "Atlanta", "ATL"],
            "Boston Celtics": ["Celtics", "Boston", "BOS"],
            "Brooklyn Nets": ["Nets", "Brooklyn", "BKN"],
            "Charlotte Hornets": ["Hornets", "Charlotte", "CHA"],
            "Chicago Bulls": ["Bulls", "Chicago", "CHI"],
            "Cleveland Cavaliers": ["Cavaliers", "Cleveland", "CLE"],
            "Dallas Mavericks": ["Mavericks", "Dallas", "DAL"],
            "Denver Nuggets": ["Nuggets", "Denver", "DEN"],
            "Detroit Pistons": ["Pistons", "Detroit", "DET"],
            "Golden State Warriors": ["Warriors", "Golden State", "GSW"],
            "Houston Rockets": ["Rockets", "Houston", "HOU"],
            "Indiana Pacers": ["Pacers", "Indiana", "IND"],
            "Los Angeles Clippers": ["Clippers", "LA Clippers", "LAC"],
            "Los Angeles Lakers": ["Lakers", "LA Lakers", "LAL"],
            "Memphis Grizzlies": ["Grizzlies", "Memphis", "MEM"],
            "Miami Heat": ["Heat", "Miami", "MIA"],
            "Milwaukee Bucks": ["Bucks", "Milwaukee", "MIL"],
            "Minnesota Timberwolves": ["Timberwolves", "Minnesota", "MIN"],
            "New Orleans Pelicans": ["Pelicans", "New Orleans", "NOP"],
            "New York Knicks": ["Knicks", "New York", "NYK"],
            "Oklahoma City Thunder": ["Thunder", "Oklahoma City", "OKC"],
            "Orlando Magic": ["Magic", "Orlando", "ORL"],
            "Philadelphia 76ers": ["76ers", "Philadelphia", "PHI"],
            "Phoenix Suns": ["Suns", "Phoenix", "PHX"],
            "Portland Trail Blazers": ["Trail Blazers", "Portland", "POR"],
            "Sacramento Kings": ["Kings", "Sacramento", "SAC"],
            "San Antonio Spurs": ["Spurs", "San Antonio", "SAS"],
            "Toronto Raptors": ["Raptors", "Toronto", "TOR"],
            "Utah Jazz": ["Jazz", "Utah", "UTA"],
            "Washington Wizards": ["Wizards", "Washington", "WAS"]
        }
        
        self.soccer_leagues = ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1"]
        self.soccer_teams = ["Nott\'ham Forest", "Alavés", "Almería", "Arsenal", "Aston Villa", "Athletic Club", "Atlético Madrid", "Barcelona", "Betis", "Bournemouth", "Brentford", "Brest", "Brighton", "Burnley", "Celta Vigo", "Chelsea", "Clermont Foot", "Crystal Palace", "Cádiz", "Everton", "Fulham", "Getafe", "Girona", "Granada", "Las Palmas", "Le Havre", "Lens", "Lille", "Liverpool", "Lorient", "Luton Town", "Lyon", "Mallorca", "Manchester City", "Manchester Utd", "Marseille", "Metz", "Monaco", "Montpellier", "Nantes", "Newcastle Utd", "Nice", "Osasuna", "Paris S-G", "Rayo Vallecano", "Real Madrid", "Real Sociedad", "Reims", "Rennes", "Sevilla", "Sheffield Utd", "Strasbourg", "Tottenham", "Toulouse", "Valencia", "Villarreal", "West Ham", "Wolves"]
        self.soccer_teams_alter = {
            "Nott\'ham Forest": ["Nottham Forest"],
            "Alavés": [],
            "Almería": [],
            "Arsenal": [],
            "Aston Villa": [],
            "Athletic Club": [],
            "Atlético Madrid": [],
            "Barcelona": [],
            "Betis": [],
            "Bournemouth": [],
            "Brentford": [],
            "Brest": [],
            "Brighton": [],
            "Burnley": [],
            "Celta Vigo": [],
            "Chelsea": [],
            "Clermont Foot": [],
            "Crystal Palace": [],
            "Cádiz": [],
            "Everton": [],
            "Fulham": [],
            "Getafe": [],
            "Girona": [],
            "Granada": [],
            "Las Palmas": [],
            "Le Havre": [],
            "Lens": [],
            "Lille": [],
            "Liverpool": [],
            "Lorient": [],
            "Luton Town": [],
            "Lyon": [],
            "Mallorca": [],
            "Manchester City": [],
            "Manchester Utd": [],
            "Marseille": [],
            "Metz": [],
            "Monaco": [],
            "Montpellier": [],
            "Nantes": [],
            "Newcastle Utd": [],
            "Nice": [],
            "Osasuna": [],
            "Paris S-G": [],
            "Rayo Vallecano": [],
            "Real Madrid": [],
            "Real Sociedad": [],
            "Reims": [],
            "Rennes": [],
            "Sevilla": [],
            "Sheffield Utd": [],
            "Strasbourg": [],
            "Tottenham": [],
            "Toulouse": [],
            "Valencia": [],
            "Villarreal": [],
            "West Ham": [],
            "Wolves": []
        }
        
    def get_nba_teams(self, query):
        teams = []
        for team in self.nba_teams:
            if query.lower() in team.lower():
                teams.append(team)
            else:
                for team_alt in self.nba_teams_alter[team]:
                    for q in query.split():
                        if q.lower() == team_alt.lower():
                            teams.append(team)
        return list(set(teams))
    
    def get_soccer_teams(self, query):
        teams = []
        for team in self.soccer_teams:
            if query.lower() in team.lower():
                teams.append(team)
            else:
                for team_alt in self.soccer_teams_alter[team]:
                    for q in query.split():
                        if q.lower() == team_alt.lower():
                            teams.append(team)
        return teams
    
    def get_soccer_leagues(self, query):
        leagues = []
        for league in self.soccer_leagues:
            if query.lower() in league.lower():
                leagues.append(league)
        return leagues
    
    def soccer_get_games_on_date(self, date, soccer_team_name):
        games = self.api.soccer_games_by_date(date, soccer_team_name)['result']
        games_ = {}
        games = json.loads(games)
        keys = games.keys()
        keys_ = games['date'].keys()
        for key in keys_:
            games_[key] = {}
            for k in keys:
                games_[key][k] = games[k][key]
                if k == 'date':
                    games_[key][k] = games_[key][k][:10]
        return games_
    
    def get_nba_games_by_date(self, date, basketball_team_name):
        return self.api.nba_games_by_date(date, basketball_team_name)['result']
    
    def get_nba_play_by_play(self, game_ids):
        return self.api.nba_play_by_play(game_ids)['reult']

