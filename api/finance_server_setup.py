import os
import string
import numpy as np
import pandas as pd
from utils import logger
from fastapi import APIRouter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sqlitedict import SqliteDict

KG_BASE_DIRECTORY = os.getenv("KG_BASE_DIRECTORY", "mock_api/cragkg")
class FinanceKG():
    def __init__(self):
        self.fuzzy_n = 10
        company_dict_file_path = os.path.join(KG_BASE_DIRECTORY, "finance", 'company_name.dict')
        logger.info("Reading company name")
        df = pd.read_csv(company_dict_file_path)[["Name", "Symbol"]]
        self.name_dict = dict(df.values)

        self.key_map = dict()
        self.corpus = []
        for e in self.name_dict:
            ne = self.normalize(e)
            if ne not in self.key_map:
                self.key_map[ne] = []
            self.key_map[ne].append(e)
            self.corpus.append(ne.split())
        self.bm25 = BM25Okapi(self.corpus)
        self._load_db()
    
    def _load_db(self):
        price_history_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_price.sqlite")
        logger.info(f"Reading price history from: {price_history_path}")
        self.price_history = SqliteDict(price_history_path)

        detailed_price_history_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_detailed_price.sqlite")
        logger.info(f"Reading detailed price history from: {detailed_price_history_path}")
        self.detailed_price_history = SqliteDict(detailed_price_history_path)

        dividend_history_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_dividend.sqlite")
        logger.info(f"Reading dividend history from: {dividend_history_path}")
        self.dividend_history = SqliteDict(dividend_history_path)

        market_cap_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_marketcap.sqlite")
        logger.info(f"Reading market capitalization from: {market_cap_path}")
        self.market_cap = SqliteDict(market_cap_path)

        financial_info_path = os.path.join(KG_BASE_DIRECTORY, "finance", "finance_info.sqlite")
        logger.info(f"Reading financial information from: {financial_info_path}")
        self.financial_info = SqliteDict(financial_info_path)
        
    
    def normalize(self, x:str) -> str:
        return " ".join(x.lower().replace("_", " ").translate(str.maketrans('', '', string.punctuation)).split())

    def get_company_name(self, query:str) -> list[str]:
        query = self.normalize(query)
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:self.fuzzy_n]
        top_ne = [" ".join(self.corpus[i]) for i in top_idx if scores[i] != 0]
        top_e = []
        for ne in top_ne:
            assert(ne in self.key_map)
            top_e += self.key_map[ne]
        return top_e[:self.fuzzy_n]

    def get_ticker_by_name(self, company_name:str) -> str:
        return self.name_dict.get(company_name, None)

    def get_price_history(self, ticker_name:str):
        db = self.price_history
        if ticker_name in db:
            return db[ticker_name]

    def get_detailed_price_history(self, ticker_name:str):
        db = self.detailed_price_history
        if ticker_name in db:
            return db[ticker_name]

    def get_dividends_history(self, ticker_name:str):
        db = self.dividend_history
        if ticker_name in db:
            return db[ticker_name]

    def get_market_capitalization(self, ticker_name: str) -> float:
        db = self.market_cap
        if ticker_name in db:
            return db[ticker_name]

    def get_eps(self, ticker_name:str) -> float:
        db = self.financial_info
        if ticker_name in db and 'forwardEps' in db[ticker_name]:
            return db[ticker_name]['forwardEps']

    def get_pe_ratio(self, ticker_name:str) -> float:
        db = self.financial_info
        if ticker_name in db and 'forwardPE' in db[ticker_name]:
            return db[ticker_name]['forwardPE']
    
    def get_info(self, ticker_name:str):
        db = self.financial_info
        if ticker_name in db:
            return db[ticker_name]

router = APIRouter()
finance_api = FinanceKG()

class CompanyRequest(BaseModel):
    company_name: str

class TickerRequest(BaseModel):
    ticker_name: str

@router.post('/finance/get_company_name')
def search_company_name(req: CompanyRequest):
    result = finance_api.get_company_name(req.company_name)
    return {'result': result}

@router.post('/finance/get_ticker_by_name')
def search_ticker_name(req: CompanyRequest):
    result = finance_api.get_ticker_by_name(req.company_name)
    return {'result': result}

@router.post('/finance/get_price_history')
def search_price(req: TickerRequest):
    result = finance_api.get_price_history(req.ticker_name)
    return {'result': result}

@router.post('/finance/get_detailed_price_history')
def search_detailed_price(req: TickerRequest):
    result = finance_api.get_detailed_price_history(req.ticker_name)
    return {'result': result}

@router.post('/finance/get_dividends_history')
def search_dividends(req: TickerRequest):
    result = finance_api.get_dividends_history(req.ticker_name)
    return {'result': result}

@router.post('/finance/get_market_capitalization')
def search_market_cap(req: TickerRequest):
    result = finance_api.get_market_capitalization(req.ticker_name)
    return {'result': result}

@router.post('/finance/get_eps')
def search_eps(req: TickerRequest):
    result = finance_api.get_eps(req.ticker_name)
    return {'result': result}

@router.post('/finance/get_pe_ratio')
def search_pe_ratio(req: TickerRequest):
    result = finance_api.get_pe_ratio(req.ticker_name)
    return {'result': result}

@router.post('/finance/get_info')
def search_get_info(req: TickerRequest):
    result = finance_api.get_info(req.ticker_name)
    return {'result': result}

#%%
