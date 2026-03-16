import os
import pickle
import string
import pandas as pd
from utils import logger
from .fast_bm25 import BM25
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "mock_api/cragkg")
class MusicKG(object):
    def __init__(self):
        artist_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_dict_simplified.pickle")
        logger.info(f"Reading artist dictionary from: {artist_dict_path}")
        with open(artist_dict_path, 'rb') as file:
            self.artist_dict = pickle.load(file)

        song_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_simplified.pickle")
        logger.info(f"Reading song dictionary from: {song_dict_path}")
        with open(song_dict_path, 'rb') as file:
            self.song_dict = pickle.load(file)

        grammy_df_path = os.path.join(KG_BASE_DIRECTORY, "music", "grammy_df.csv")
        logger.info(f"Reading Grammy DataFrame from: {grammy_df_path}")
        self.grammy_df = pd.read_csv(grammy_df_path)

        rank_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "rank_dict_hot100.pickle")
        logger.info(f"Reading rank dictionary for Hot 100 from: {rank_dict_hot_path}")
        with open(rank_dict_hot_path, 'rb') as file:
            self.rank_dict_hot = pickle.load(file)

        song_dict_hot_path = os.path.join(KG_BASE_DIRECTORY, "music", "song_dict_hot100.pickle")
        logger.info(f"Reading song dictionary for Hot 100 from: {song_dict_hot_path}")
        with open(song_dict_hot_path, 'rb') as file:
            self.song_dict_hot = pickle.load(file)

        artist_work_dict_path = os.path.join(KG_BASE_DIRECTORY, "music", "artist_work_dict.pickle")
        logger.info(f"Reading artist work dictionary from: {artist_work_dict_path}")
        with open(artist_work_dict_path, 'rb') as file:
            self.artist_work_dict = pickle.load(file)
        
        self.key_map_artist = {}
        self.corpus_artist = []
        for e in self.artist_dict.keys():
            ne = self.normalize(e)
            ne_split = str(ne.split())
            if ne_split not in self.key_map_artist:
                self.key_map_artist[ne_split] = []
            self.key_map_artist[ne_split].append(e)
            self.corpus_artist.append(ne)
        self.corpus_artist = list(set(self.corpus_artist))
        self.corpus_artist.sort()
        self.corpus_artist = [ne.split() for ne in self.corpus_artist]
        self.bm25_artist = BM25(self.corpus_artist)

        self.key_map_song = {}
        self.corpus_song = []
        for e in self.song_dict.keys():
            ne = self.normalize(e)
            ne_split = str(ne.split())
            if ne_split not in self.key_map_song:
                self.key_map_song[ne_split] = []
            self.key_map_song[ne_split].append(e)
            self.corpus_song.append(ne)
        self.corpus_song = list(set(self.corpus_song))
        self.corpus_song.sort()
        self.corpus_song = [ne.split() for ne in self.corpus_song]
        self.bm25_song = BM25(self.corpus_song)
    
    def normalize(self, x):
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())
    
    def search_artist_entity_by_name(self, query):
        n = 10
        query = self.normalize(query)
        results = self.bm25_artist.get_top_n(query.split(), self.corpus_artist, n=n)
        top_e = []
        for cur_ne_str in results:
            assert(str(cur_ne_str) in self.key_map_artist.keys())
            top_e += self.key_map_artist[str(cur_ne_str)]
        return top_e[:n]
    
    def search_song_entity_by_name(self, query):
        n = 10
        query = self.normalize(query)
        results = self.bm25_song.get_top_n(query.split(), self.corpus_song, n=n)
        top_e = []
        for cur_ne_str in results:
            assert(str(cur_ne_str) in self.key_map_song.keys())
            top_e += self.key_map_song[str(cur_ne_str)]
        return top_e[:n]

    def get_billboard_rank_date(self, rank, date=None):
        rank_list = []
        artist_list = []        
        if not str(rank) in self.rank_dict_hot.keys():
            return None, None
        else:
            if date:
                for item in self.rank_dict_hot[str(rank)]:
                    if item['Date'] == date:
                        return [item['Song']], [item['Artist']]
            else:
                for item in self.rank_dict_hot[str(rank)]:
                    rank_list.append(item['Song'])
                    artist_list.append(item['Artist'])
        return rank_list, artist_list
        
    def get_billboard_attributes(self, date, attribute, song_name):
        if not song_name in self.song_dict_hot:
            return None
        else:
            cur_dict = self.song_dict_hot[song_name]
            if not date in cur_dict.keys():
                return None
            else:
                row = cur_dict[date]
                if row[6] == '-':
                    if attribute == 'rank_last_week':
                        cur_value = row[6]
                    elif attribute == 'weeks_in_chart':
                        cur_value = row[5]
                    elif attribute == 'top_position':
                        cur_value = row[4]
                    else:
                        cur_value = row[3]
                else:
                    if attribute == 'rank_last_week':
                        cur_value = row[4]
                    elif attribute == 'weeks_in_chart':
                        cur_value = row[6]
                    elif attribute == 'top_position':
                        cur_value = row[5]
                    else:
                        cur_value = row[3]
                return cur_value
    
    def grammy_get_best_artist_by_year(self, year):
        if year<1957 or year>2019:
            return None
        else:
            filtered_df = self.grammy_df[(self.grammy_df['category'] == 'Best New Artist') & (self.grammy_df['year'] == year)]
            artist_list = filtered_df['nominee'].tolist()
            return artist_list
    
    def grammy_get_award_count_by_artist(self, artist_name):
        total_unique_rows_artist = 0
        total_unique_rows_nominee = 0
        total_unique_rows_worker = 0
        for value in self.grammy_df['nominee']:
            if artist_name in str(value):
                total_unique_rows_nominee += 1
        for value in self.grammy_df['artist']:
            if artist_name in str(value):
                total_unique_rows_artist += 1
        for value in self.grammy_df['workers']:
            if artist_name in str(value):
                total_unique_rows_worker += 1
        return total_unique_rows_nominee + total_unique_rows_artist + total_unique_rows_worker
    
    def grammy_get_award_count_by_song(self, song_name):
        total_unique_rows_nominee = len(self.grammy_df[self.grammy_df['nominee']==song_name])
        return total_unique_rows_nominee
    
    def grammy_get_best_song_by_year(self, year):
        if year<1957 or year>2019:
            return None
        else:
            filtered_df = self.grammy_df[(self.grammy_df['category'] == 'Song Of The Year') & (self.grammy_df['year'] == year)]
            song_list = filtered_df['nominee'].tolist()
            return song_list
    
    def grammy_get_award_date_by_artist(self, artist_name):
        idx = []
        for i, value in enumerate(self.grammy_df['nominee']):
            if artist_name in str(value):
                idx.append(i)
        for i, value in enumerate(self.grammy_df['artist']):
            if artist_name in str(value):
                idx.append(i)
        for i, value in enumerate(self.grammy_df['workers']):
            if artist_name in str(value):
                idx.append(i)
        selected_idx = list(set(idx))
        selected_years = []
        for cur_idx in selected_idx:
            selected_years.append(self.grammy_df['year'][cur_idx])
        selected_years = list(set(selected_years))
        selected_years = [int(x) for x in selected_years]
        return selected_years
    
    def grammy_get_best_album_by_year(self, year):
        if year<1957 or year>2019:
            return None
        else:
            filtered_df = self.grammy_df[(self.grammy_df['category'] == 'Album Of The Year') & (self.grammy_df['year'] == year)]
            song_list = filtered_df['nominee'].tolist()
            return song_list
    
    def grammy_get_all_awarded_artists(self):
        nominee_values = self.grammy_df[self.grammy_df['category'] == 'Best New Artist']['nominee'].dropna().unique().tolist()
        return nominee_values
    
    def get_artist_birth_place(self, artist_name):
        try:
            d = self.artist_dict[artist_name]
            country = d['country']
            if country:
                return country
            else:
                return None
        except:
            return None
    
    def get_artist_birth_date(self, artist_name):
        try:
            d = self.artist_dict[artist_name]
            life_span_begin = d['birth_date']
            if life_span_begin:
                return life_span_begin
            else:
                return None
        except:
            return None
    
    def get_members(self, band_name):
        try:
            d = self.artist_dict[band_name]
            members = d['members']
            return list(set(members))
        except:
            return None
    
    def get_lifespan(self, artist_name):
        try:
            d = self.artist_dict[artist_name]
            life_span_begin = d['birth_date']
            life_span_end = d['end_date']
            life = [life_span_begin, life_span_end]
            return life
        except:
            return [None, None]
    
    def get_song_author(self, song_name):
        try:
            d = self.song_dict[song_name]
            author = d['author']
            if author:
                return author
            else:
                return None
        except:
            return None
    
    def get_song_release_country(self, song_name):
        try:
            d = self.song_dict[song_name]
            country = d['country']
            if country:
                return country
            else:
                return None
        except:
            return None
    
    def get_song_release_date(self, song_name):
        try:
            d = self.song_dict[song_name]
            date = d['date']
            if date:
                return date
            else:
                return None
        except:
            return None
    
    def get_artist_all_works(self, artist_name):
        if artist_name in self.artist_work_dict.keys():
            work_list = self.artist_work_dict[artist_name]
            return work_list
        else:
            return None

router = APIRouter()
music_api = MusicKG()

class QueryRequest(BaseModel):
    query: str

class RankRequest(BaseModel):
    rank: int
    date: Optional[str] = None

class BillBoardRequest(BaseModel):
    date: str
    attribute: str
    song_name: str

class YearRequest(BaseModel):
    year: int

class ArtistRequest(BaseModel):
    artist_name: str

class SongRequest(BaseModel):
    song_name: str

class BandRequest(BaseModel):
    band_name: str

@router.post('/music/search_artist_entity_by_name')
def search_artist_name(req: QueryRequest):
    result = music_api.search_artist_entity_by_name(req.query)
    return {'result': result}

@router.post('/music/search_song_entity_by_name')
def search_song_name(req: QueryRequest):
    result = music_api.search_song_entity_by_name(req.query)
    return {'result': result}

@router.post('/music/get_billboard_rank_date')
def search_billboard_rank(req: RankRequest):
    result = music_api.get_billboard_rank_date(req.rank, req.date)
    return {'result': result}

@router.post('/music/get_billboard_attributes')
def search_billboard_attribute(req: BillBoardRequest):
    result = music_api.get_billboard_attributes(req.date, req.attribute, req.song_name)
    return {'result': result}

@router.post('/music/grammy_get_best_artist_by_year')
def search_grammy_artist(req: YearRequest):
    result = music_api.grammy_get_best_artist_by_year(req.year)
    return {'result': result}

@router.post('/music/grammy_get_award_count_by_artist')
def search_grammy_artist_award_count(req: ArtistRequest):
    result = music_api.grammy_get_award_count_by_artist(req.artist_name)
    return {'result': result}

@router.post('/music/grammy_get_award_count_by_song')
def search_grammy_song_award_count(req: SongRequest):
    result = music_api.grammy_get_award_count_by_song(req.song_name)
    return {'result': result}

@router.post('/music/grammy_get_best_song_by_year')
def search_best_song(req: YearRequest):
    result = music_api.grammy_get_best_song_by_year(req.year)
    return {'result': result}

@router.post('/music/grammy_get_award_date_by_artist')
def search_artist_award(req: ArtistRequest):
    result = music_api.grammy_get_award_date_by_artist(req.artist_name)
    return {'result': result}

@router.post('/music/grammy_get_best_album_by_year')
def search_best_album(req: YearRequest):
    result = music_api.grammy_get_best_album_by_year(req.year)
    return {'result': result}

@router.post('/music/grammy_get_all_awarded_artists')
def search_artist_all_awards():
    result = music_api.grammy_get_all_awarded_artists()
    return {'result': result}

@router.post('/music/get_artist_birth_place')
def search_artist_birth_place(req: ArtistRequest):
    result = music_api.get_artist_birth_place(req.artist_name)
    return {'result': result}

@router.post('/music/get_artist_birth_date')
def search_artist_birth_date(req: ArtistRequest):
    result = music_api.get_artist_birth_date(req.artist_name)
    return {'result': result}

@router.post('/music/get_members')
def search_band_members(req: BandRequest):
    result = music_api.get_members(req.band_name)
    return {'result': result}

@router.post('/music/get_lifespan')
def search_lifespan(req: ArtistRequest):
    result = music_api.get_lifespan(req.artist_name)
    return {'result': result}

@router.post('/music/get_song_author')
def search_author(req: SongRequest):
    result = music_api.get_song_author(req.song_name)
    return {'result': result}

@router.post('/music/get_song_release_country')
def search_song_country(req: SongRequest):
    result = music_api.get_song_release_country(req.song_name)
    return {'result': result}

@router.post('/music/get_song_release_date')
def search_song_release_date(req: SongRequest):
    result = music_api.get_song_release_date(req.song_name)
    return {'result': result}

@router.post('/music/get_artist_all_works')
def search_artist_works(req: ArtistRequest):
    result = music_api.get_artist_all_works(req.artist_name)
    return {'result': result}


#%%