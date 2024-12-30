import os

from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG = os.environ.get("DEBUG", bool(0))
API_KEY_OPENWEATHER = os.environ.get("API_KEY_OPENWHEATER")

def getwheater():
    path_file = os.path.join(CURRENT_DIR, "weather_last7days.json")
    with open(path_file, "r") as f:
        return f.read()
    return None

DATA_WEATHER = getwheater()
