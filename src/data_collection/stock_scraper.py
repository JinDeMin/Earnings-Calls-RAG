import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

class StockScraper:
    def __init__(self):
        self.url = "https://stockanalysis.com/stocks/"

    def get_stock_data(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, "html.parser")
        script = soup.find_all("script")[-1].text
        json_data = self._extract_json_data(script)
        return self._create_dataframe(json_data)

    def _extract_json_data(self, script):
        javascript_json_string = script.split(';')[2].split('=')[1]
        javascript_json_string = javascript_json_string.split('{\"type\":\"data\",\"data\":{data:')[1]
        javascript_json_string = javascript_json_string.split('}]')[0]+'}]'
        javascript_json_string = javascript_json_string.replace('s:','\"s\":').replace(',n:',',\"n\":').replace(',i:',',\"i\":').replace(',m:',',\"m\":')
        return json.loads(javascript_json_string)

    def _create_dataframe(self, data):
        df = pd.DataFrame(data)
        df.columns = ['Ticker', 'Name', 'Industry', 'Market_Cap']
        return df

    def get_filtered_stocks(self, filter_names):
        df = self.get_stock_data()
        return df[df['Name'].isin(filter_names)]