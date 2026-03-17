import os
import pandas as pd
import sqlite3 as sql
from utils import logger
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "mock_api/cragkg")

class SoccerKG:
    def __init__(self, file_name='soccer_team_match_stats.pkl'):
        soccer_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", file_name)
        logger.info(f"Reading soccer KG from: {soccer_kg_file}")
        team_match_stats = pd.read_pickle(os.path.join(KG_BASE_DIRECTORY, "sports", file_name))
        self.team_match_stats = team_match_stats[team_match_stats.index.get_level_values('league').notna()]
        logger.info("Soccer KG initialized")
        
    def get_games_on_date(self, date_str, soccer_team_name=None):
        parts = date_str.split('-')
        if soccer_team_name is None:
            filtered_df = self.team_match_stats
        else:
            filtered_df = self.team_match_stats.loc[(slice(None), slice(None), soccer_team_name, slice(None)), :]
        if len(parts) == 3:
            filtered_df = filtered_df[filtered_df['date'].dt.strftime('%Y-%m-%d') == date_str]    
        elif len(parts) == 2: 
            filtered_df = filtered_df[filtered_df['date'].dt.strftime('%Y-%m') == date_str]
        elif len(parts) == 1:
            filtered_df = filtered_df[filtered_df['date'].dt.strftime('%Y') == date_str]
        else:
            filtered_df = None
        if filtered_df is not None and len(filtered_df) > 0:
            return filtered_df.to_json(date_format='iso')

class NBAKG:
    def __init__(self):
        nba_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", 'nba.sqlite')
        logger.info(f"Reading NBA KG from: {nba_kg_file}")
        self.conn = sql.connect(nba_kg_file)

    def get_time_cond(self, date_str):
        parts = date_str.split('-')
        if len(parts) == 3:
            return f"strftime('%Y-%m-%d',game_date) = '{date_str}'"
        elif len(parts) == 2: 
            return f"strftime('%Y-%m',game_date) = '{date_str}'"
        elif len(parts) == 1:
            return f"strftime('%Y',game_date) = '{date_str}'"
        else:
            return "1"
    
    def team_in_game_cond(self, basketball_team_name):
        return f"(team_name_home = '{basketball_team_name}' or team_name_away = '{basketball_team_name}')"

    def get_games_on_date(self, date_str, basketball_team_name=None):
        if basketball_team_name is not None:
            team_cond = self.team_in_game_cond(basketball_team_name)
            time_cond = self.get_time_cond(date_str)
            df_game_by_team = pd.read_sql(f"select * from game where {team_cond} and {time_cond}", self.conn)
            if len(df_game_by_team) > 0:
                return df_game_by_team.to_json(date_format='iso')
        else:
            time_cond = self.get_time_cond(date_str)
            df_game_by_team = pd.read_sql(f"select * from game where {time_cond}", self.conn)
            if len(df_game_by_team) > 0:
                return df_game_by_team.to_json(date_format='iso')

    def get_play_by_play_data_by_game_ids(self, game_ids):
        game_ids_str = ', '.join(f"'{game_id}'" for game_id in game_ids)
        df_play_by_play_by_gameids = pd.read_sql(f"select * from play_by_play where game_id in ({game_ids_str})", self.conn)
        if len(df_play_by_play_by_gameids) > 0:
            return df_play_by_play_by_gameids.to_json(date_format='iso')   

router = APIRouter()
soccer_api = SoccerKG()
nba_api = NBAKG()

class GamesOnDate(BaseModel):
    date_str: str
    soccer_team_name: Optional[str] = None

class GameIds(BaseModel):
    game_ids: str

@router.post('/sports/get_games_on_date')
def search_soccer_gaames(req: GamesOnDate):
    result = soccer_api.get_games_on_date(req.date_str, req.soccer_team_name)
    return {'result': result}

@router.post('/sports/get_games_on_date')
def search_nba_gaames(req: GamesOnDate):
    result = nba_api.get_games_on_date(req.date_str, req.basketball_team_name)
    return {'result': result}

@router.post('/sports/get_play_by_play_data_by_game_ids')
def search_nba_play_by_play(req: GameIds):
    result = nba_api.get_play_by_play_data_by_game_ids(req.game_ids)
    return {'result': result}


#%%