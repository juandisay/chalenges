import random

from typing import List, Dict, Any
from shared import Requests
from sohoo.list_country import search
from shared.settings import DEBUG, API_KEY_OPENWEATHER, DATA_WEATHER


class OpenWeatherProvider:
    def __init__(self):
        self.client = Requests()
        self.random_api_key = random.choice(API_KEY_OPENWEATHER.split(","))

    def _current_weather(self, search_cities) -> List[Dict[str, Any]]:
        """
        Current weather per cities

        Args:
          - search_cities: name of city search for
        Returns:
          List of Weather per cities
        """

        result = []
        cities = search(search_cities)
        for loc in cities:
            API_URL = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={loc.get('coord')['lat']}&lon={loc.get('coord')['lon']}"
                f"&appid={self.random_api_key}"
            )

            resp = self.client.get(API_URL)
            result.append(resp.json())
        return result


class OpenMateoProvider:
    def __init__(self):
        self.client = Requests()

    def _last7days(self, search_cities):
        """
        Last 7 days history weather per cities

        Args:
          - search_cities: name of city search for
        Returns:
          List of Weather per cities
        """
        result = []
        cities = search(search_cities)
        for loc in cities:
            lat = loc.get('coord')['lat']
            lon = loc.get('coord')['lon']
            API_URL = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&past_days=7&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
            )

            if DEBUG:
                print(API_URL)
            
            resp = self.client.get(API_URL)
            result = resp.json()
        return result

get_openWeather = OpenWeatherProvider()
get_openMateo = OpenMateoProvider()
