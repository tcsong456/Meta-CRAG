import os
import json
import string
import numpy as np
from fastapi import APIRouter
from rank_bm25 import BM25Okapi
from pydantic import BaseModel

class MovieKG:
    def __init__(self, top_n):
        KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "cragkg")
        
        year_db_path = os.path.join(KG_BASE_DIRECTORY, "movie", "year_db.json")
        with open(year_db_path) as f:
            self._year_db = json.load(f)
        
        person_db_path = os.path.join(KG_BASE_DIRECTORY, "movie", "person_db.json")
        with open(person_db_path) as f:
            self._person_db = json.load(f)
            
        movie_db_path = os.path.join(KG_BASE_DIRECTORY, "movie", "movie_db.json")
        with open(movie_db_path) as f:
            self._movie_db = json.load(f)
        
        self.top_n = top_n
        self._person_db_lookup = self._get_direct_lookup_db(self._person_db)
        self._movie_db_lookup = self._get_direct_lookup_db(self._movie_db)
        self._movie_corpus, self._movie_bm25 = self._get_ranking_db(self._movie_db)
        self._person_corpus, self._person_bm25 = self._get_ranking_db(self._person_db)
    
    def _get_direct_lookup_db(self, db):
        temp_db = {}    
        for key, value in db.items():
            if 'id' in value:
                temp_db[value['id']] = value
        return temp_db
    
    def _get_ranking_db(self, db) :
        corpus = [i.split() for i in db.keys()]
        bm25 = BM25Okapi(corpus)
        return corpus, bm25
    
    def _normalize(self, x):
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())
    
    def _search_entity_by_name(self, query, bm25, corpus, map_db):
        n = self._top_n
        query = self._normalize(query)
        scores = bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:n]
        top_ne = [" ".join(corpus[i]) for i in top_idx if scores[i] != 0]
        top_e = []
        for ne in top_ne[:n]:
            assert(ne in map_db)
            top_e.append(map_db[ne])
        return top_e[:n]
    
    def get_person_info(self, person_name):
        res = self._search_entity_by_name(person_name, self._person_bm25, self._person_corpus, self._person_db)
        return res
    
    def get_movie_info(self, movie_name):
        res = self._search_entity_by_name(movie_name, self._movie_bm25, self._movie_corpus, self._movie_db)
        return res
    
    def get_movie_info_by_id(self, movie_id):
        return self._movie_db_lookup.get(movie_id, None)
    
    def get_person_info_by_id(self, person_id):
        return self._person_db_lookup.get(person_id, None)
    
    def get_year_info(self, year):
        if int(year) not in range(1990, 2022):
            raise ValueError("Year must be between 1990 and 2021")
        return self._year_db.get(str(year), None)

router = APIRouter()
movie_api = MovieKG()

class PersonRequest(BaseModel):
    person: str

class MovieRequest(BaseModel):
    movie: str

class YearRequest(BaseModel):
    year: str

class MovieIdRequest(BaseModel):
    movie_id: int

class PersonIdRequest(BaseModel):
    person_id: int

@router.post('/movie/get_person_info')
def search_person_info(req: PersonRequest):
    result = movie_api.get_person_info(req.person)
    return {'result': result}

@router.post('/movie/get_movie_info')
def search_movie_info(req: MovieRequest):
    result = movie_api.get_movie_info(req.movie)
    return {'result': result}

@router.post('/movie/get_year_info')
def search_year_info(req: YearRequest):
    result = movie_api.get_movie_info(req.year)
    return {'result': result}

@router.post('/movie/get_movie_info_by_id')
def search_movie_info_by_id(req: MovieIdRequest):
    result = movie_api.get_movie_info_by_id(req.movie_id)
    return {'result': result}

@router.post('/movie/get_person_info_by_id')
def search_person_info_by_id(req: MovieIdRequest):
    result = movie_api.get_movie_info_by_id(req.person_id)
    return {'result': result}


#%%
