import re
import json
import requests
from bs4 import BeautifulSoup

url = 'https://www.espncricinfo.com/series/8048/commentary/1181768/mumbai-indians-vs-chennai-super-kings-final-indian-premier-league-2019'
api_url = 'https://hsapi.espncricinfo.com/v1/pages/match/comments?lang=en&leagueId={leagueId}&eventId={eventId}&liveTest=false&filter=full&page={page}'

leagueId, eventId = re.findall(r'(\d+)/commentary/(\d+)', url)[0]

page = 1
while True:
    data = requests.get(api_url.format(page=page, leagueId=leagueId, eventId=eventId)).json()

    # uncomment next line to see all data:
    # print(json.dumps(data, indent=4))

    # print some data to screen:
    for comment in data['comments']:
        soup1 = BeautifulSoup(comment['preText'], 'html.parser')
        soup2 = BeautifulSoup(comment['text'], 'html.parser')
        soup3 = BeautifulSoup(comment['postText'], 'html.parser')

        print(soup1.get_text(strip=True, separator='\n'))
        print(soup2.get_text(strip=True, separator='\n'))
        print(soup3.get_text(strip=True, separator='\n'))

        print('-' * 80)

    page += 1

    if page > data['pagination']['pageCount']:
        break