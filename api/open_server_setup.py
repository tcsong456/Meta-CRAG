import os
import bz2
import json
import string
import numpy as np
from utils import logger
from fastapi import APIRouter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

class OpenKG(object):
    def __init__(self):
        self.kg = {}
        for i in range(2):
            open_kg_file = os.path.join('mock_api/cragkg', "open", "kg."+str(i)+".jsonl.bz2")
            logger.info(f"Reading open_kg file from: {open_kg_file}")
            with bz2.open(open_kg_file, "rt", encoding='utf8') as f:
                l = f.readline()
                while l:
                    l = json.loads(l)
                    self.kg[l[0]] = l[1]
                    l = f.readline()
        self.key_map = {}
        self.corpus = []
        for e in self.kg:
            ne = self.normalize(e)
            if ne not in self.key_map:
                self.key_map[ne] = []
            self.key_map[ne].append(e)
            self.corpus.append(ne)
        self.corpus = list(set(self.corpus))
        self.corpus.sort()
        self.corpus = [ne.split() for ne in self.corpus]
        self.bm25 = BM25Okapi(self.corpus)
        
        logger.info("Open KG initialized")

        
    def normalize(self, x):
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())

    def search_entity_by_name(self, query):
        n = 10
        query = self.normalize(query)
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:n]
        top_ne = [" ".join(self.corpus[i]) for i in top_idx if scores[i] != 0]
        top_e = []
        for ne in top_ne:
            assert(ne in self.key_map)
            top_e += self.key_map[ne]
        return top_e[:n]

    def get_entity(self, entity):
        return self.kg[entity] if entity in self.kg else None   

class QueryRequest(BaseModel):
    query: str

class EntityRequest(BaseModel):
    entity: str
    
router = APIRouter()
open_api = OpenKG()
@router.post('/open/search_entity_by_name')
def search_entity(req: QueryRequest):
    result = open_api.search_entity_by_name(req.query)
    return {'result': result}

@router.post('/open/get_entity')
def get_entity(req: EntityRequest):
    result = open_api.get_entity(req.entity)
    return {'result': result}

#%%
