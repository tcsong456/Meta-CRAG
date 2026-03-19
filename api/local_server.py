import os
import json
import requests

class CRAG:
    def __init__(self):
        self.server = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")
    
    def open_search_entity_by_name(self, query):
        url = self.server + '/open/search_entity_by_name'
        headers={'accept': "application/json"}
        data = {'query': query}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def open_get_entity(self, entity):
        url = self.server + '/open/get_entity'
        headers={'accept': "application/json"}
        data = {'entity': entity}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def movie_search_person_info(self, person):
        url = self.server + '/movie/get_person_info'
        headers={'accept': "application/json"}
        data = {'person': person}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def movie_search_info(self, movie):
        url = self.server + '/movie/get_movie_info'
        headers={'accept': "application/json"}
        data = {'movie': movie}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def movie_year_info(self, year):
        url = self.server + '/movie/get_year_info'
        headers={'accept': "application/json"}
        data = {'year': year}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def movie_search_info_by_id(self, movie_id):
        url = self.server + '/movie/get_movie_info_by_id'
        headers={'accept': "application/json"}
        data = {'movie_id': movie_id}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
        
    def movie_search_person_info_by_id(self, person_id):
        url = self.server + '/movie/get_person_info_by_id'
        headers={'accept': "application/json"}
        data = {'person_id': person_id}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
        
    def finance_get_company_name(self, company_name):
        url = self.server + '/finance/get_company_name'
        headers={'accept': "application/json"}
        data = {'company_name': company_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_ticker_name(self, company_name):
        url = self.server + '/finance/get_ticker_by_name'
        headers={'accept': "application/json"}
        data = {'company_name': company_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_price(self, ticker_name):
        url = self.server + '/finance/get_price_history'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_detailed_price(self, ticker_name):
        url = self.server + '/finance/get_detailed_price_history'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_dividends(self, ticker_name):
        url = self.server + '/finance/get_dividends_history'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_market_cap(self, ticker_name):
        url = self.server + '/finance/get_market_capitalization'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_eps(self, ticker_name):
        url = self.server + '/finance/get_eps'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_pe_ratio(self, ticker_name):
        url = self.server + '/finance/get_pe_ratio'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def finance_get_info(self, ticker_name):
        url = self.server + '/finance/get_info'
        headers={'accept': "application/json"}
        data = {'ticker_name': ticker_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_artist_name(self, query):
        url = self.server + '/music/search_artist_entity_by_name'
        headers={'accept': "application/json"}
        data = {'query': query}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_song_name(self, query):
        url = self.server + '/music/search_song_entity_by_name'
        headers={'accept': "application/json"}
        data = {'query': query}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_billboard_rank(self, rank, date):
        url = self.server + '/music/get_billboard_rank_date'
        headers={'accept': "application/json"}
        data = {'rank': rank, 'date': date}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_billboard_attribute(self, date, attribute, song_name):
        url = self.server + '/music/get_billboard_attributes'
        headers={'accept': "application/json"}
        data = {'song_name': song_name, 'date': date, 'attribute': attribute}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_grammy_artist(self, year):
        url = self.server + '/music/grammy_get_best_artist_by_year'
        headers={'accept': "application/json"}
        data = {'year': year}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_grammy_artist_award_count(self, artist_name):
        url = self.server + '/music/grammy_get_award_count_by_artist'
        headers={'accept': "application/json"}
        data = {'artist_name': artist_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_grammy_song_award_count(self, song_name):
        url = self.server + '/music/grammy_get_award_count_by_song'
        headers={'accept': "application/json"}
        data = {'song_name': song_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_best_song(self, year):
        url = self.server + '/music/grammy_get_best_song_by_year'
        headers={'accept': "application/json"}
        data = {'year': year}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_artist_award(self, artist_name):
        url = self.server + '/music/grammy_get_award_date_by_artist'
        headers={'accept': "application/json"}
        data = {'artist_name': artist_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_best_album(self, year):
        url = self.server + '/music/grammy_get_best_album_by_year'
        headers={'accept': "application/json"}
        data = {'year': year}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_artist_all_awards(self):
        url = self.server + '/music/grammy_get_all_awarded_artists'
        headers={'accept': "application/json"}
        result = requests.post(url, json={}, headers=headers)
        return json.loads(result.text)
    
    def music_search_artist_birth_place(self, artist_name):
        url = self.server + '/music/get_artist_birth_place'
        headers={'accept': "application/json"}
        data = {'artist_name': artist_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_artist_birth_date(self, artist_name):
        url = self.server + '/music/get_artist_birth_date'
        headers={'accept': "application/json"}
        data = {'artist_name': artist_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_get_band_members(self, band_name):
        url = self.server + '/music/get_members'
        headers={'accept': "application/json"}
        data = {'band_name': band_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_life_span(self, artist_name):
        url = self.server + '/music/get_lifespan'
        headers={'accept': "application/json"}
        data = {'artist_name': artist_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_author(self, song_name):
        url = self.server + '/music/get_song_author'
        headers={'accept': "application/json"}
        data = {'song_name': song_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_song_country(self, song_name):
        url = self.server + '/music/get_song_release_country'
        headers={'accept': "application/json"}
        data = {'song_name': song_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_song_release_date(self, song_name):
        url = self.server + '/music/get_song_release_date'
        headers={'accept': "application/json"}
        data = {'song_name': song_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def music_search_artist_works(self, artist_name):
        url = self.server + '/music/get_artist_all_works'
        headers={'accept': "application/json"}
        data = {'artist_name': artist_name} 
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def soccer_games_by_date(self, date_str, soccer_team_name):
        url = self.server + '/sports/get_soccer_games_on_date'
        headers={'accept': "application/json"}
        data = {'date_str': date_str, 'soccer_team_name': soccer_team_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def nba_games_by_date(self, date_str, basketball_team_name):
        url = self.server + '/sports/get_nba_games_on_date'
        headers={'accept': "application/json"}
        data = {'date_str': date_str, 'basketball_team_name': basketball_team_name}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)
    
    def nba_play_by_play(self, game_ids):
        url = self.server + '/sports/get_play_by_play_data_by_game_ids'
        headers={'accept': "application/json"}
        data = {'game_ids': game_ids}
        result = requests.post(url, json=data, headers=headers)
        return json.loads(result.text)