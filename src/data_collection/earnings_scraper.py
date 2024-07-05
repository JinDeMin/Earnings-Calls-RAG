from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import pandas as pd

class EarningsScraper:
    def __init__(self):
        self.driver = self._setup_driver()

    def _setup_driver(self):
        service = Service(executable_path=r'/usr/bin/chromedriver')
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument("window-size=1280,800")
        driver = webdriver.Chrome(service=service, options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver

    def fetch_earnings_call_urls(self, ticker, start_year, end_year):
        urls = {}
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                url = f"https://www.google.com/search?q=fool.com+ticker%3A{ticker}+earnings+call+transcript+Q{quarter}+{year}"
                self.driver.get(url)
                time.sleep(2)  # Add delay to avoid rate limiting
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                links = soup.find_all('a', href=True)
                relevant_links = [link['href'] for link in links if f'Q{quarter} {year}' in link.text]
                if relevant_links:
                    earnings_call_link = relevant_links[0]
                    if 'url?q=' in earnings_call_link:
                        earnings_call_link = earnings_call_link.split('url?q=')[1].split('&')[0]
                    urls[(year, quarter)] = earnings_call_link
        return urls

    def fetch_transcripts(self, urls):
        transcripts = {}
        for (year, quarter), url in urls.items():
            self.driver.get(url)
            time.sleep(2)  # Add delay to avoid rate limiting
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            transcript_div = soup.find('div', class_='article-body')
            if transcript_div:
                transcripts[(year, quarter)] = transcript_div.text
        return transcripts

    def close(self):
        self.driver.quit()