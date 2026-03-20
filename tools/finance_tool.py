from api.local_server import CRAG

class FinanceTool:
    def __init__(self):
        self.api = CRAG()
        company2ticker = {}
        ticker2company = {}
        all_tickers = []
        
        with open('mock_api/cragkg/finance/company_name.dict') as f:
            for line in f:
                line = line.split(",")
                if line[1] == 'Name':
                    continue
                company2ticker[line[1].strip().lower()] = line[2].strip()
                ticker2company[line[2].strip()] = line[1].strip()
                all_tickers.append(line[2].strip())
        
        self.company2ticker = company2ticker
        self.ticker2company = ticker2company
        self.all_tickers = all_tickers
    
    def get_company_name(self, query):
        return self.api.finance_get_company_name(query)['result']
    
    def get_ticker_name(self, query):
        return self.api.finance_get_ticker_name(query)['result']
    
    def get_ticker_names(self, query):
        ticker_names = []
        company_names = self.get_company_name(query)
        if company_names is None or len(company_names) == 0:
            return []
        else:
            for company_name in company_names:
                if company_name.lower() in query.lower() or query.lower() in company_name.lower():
                    ticker_names.append(self.get_ticker_name(company_name))
        return ticker_names
    
    def get_price_history(self, ticker_name):
        return self.api.finance_get_price(ticker_name.upper())['result']
    
    def get_price_by_date(self, date, ticker_name):
        ticker_name = ticker_name.upper()
        date = date + ' 00:00:00 EST'
        if self.get_price_history(ticker_name) is None:
            return None
        if date not in self.get_price_history(ticker_name):
            return None
        return self.get_price_history(ticker_name)[date]
    
    def get_latest_price(self, date, ticker_name):
        prices = self.get_price_history(ticker_name)
        dates = list(prices.keys())
        dates.sort(reverse=True)
        for d in dates:
            if d[:10] < date:
                return prices[d]
    
    def get_detailed_price_history(self, ticker_name):
        return self.api.finance_get_detailed_price(ticker_name)['result']
    
    def get_detailed_price_by_date(self, ticker_name, date, time):
        time = date + ' ' + time + ' EST'
        detailed_price_history = self.get_detailed_price_history(ticker_name.upper())
        if time in detailed_price_history:
            return detailed_price_history[time]
        else:
            return None
        
    def get_dividends_history(self, ticker_name):
        return self.api.finance_get_dividends(ticker_name.upper())['result']
    
    def get_dividend_first_date(self, ticker_name):
        return list(self.get_dividends_history(ticker_name.upper()).keys())[0]
    
    def get_latest_dividend(self, ticker_name, date):
        dividend_history = self.get_dividends_history(ticker_name.upper())
        if len(dividend_history) > 0:
            dates = list(dividend_history.keys())
            dates.sort(reverse=True)
            for d in dates:
                if d[:10] < date:
                    return {d: dividend_history[d]}
        else:
            return None
        
    def get_dividend_by_year(self, ticker_name, year):
        dividend_history = self.get_dividends_history(ticker_name.upper())
        return {date: dividend_history[date] for date in dividend_history if date[:4] == year}
    
    def get_dividend_by_month(self, ticker_name, year, month):
        dividend_history = self.get_dividends_history(ticker_name.upper())
        return {date: dividend_history[date] for date in dividend_history if date[:7] == year + '-' + month}
    
    def get_dividend_by_date(self, ticker_name, date):
        date = date + ' 00:00:00 EST'
        dividend_history = self.get_dividends_history(ticker_name.upper())
        if date in dividend_history:
            return {date: dividend_history[date]}
        else:
            return None
        
    def get_market_capitalization(self, ticker_name):
        return self.api.finance_get_market_cap(ticker_name.upper())['result']
    
    def get_eps(self, ticker_name):
        return self.api.finance_get_eps(ticker_name.upper())['result']
    
    def get_pe_ratio(self, ticker_name):
        return self.api.finance_get_pe_ratio(ticker_name.upper())['result']
    
    def get_info_keys(self, ticker_name):
        return list(self.api.finance_get_info(ticker_name.upper())['result'].keys())
    
    def get_info(self, ticker_name, key):
        return self.api.finance_get_info(ticker_name.upper())['result'][key]
    
    def get_all_info(self, ticker_name):
        return self.api.finance_get_info(ticker_name.upper())['result']
