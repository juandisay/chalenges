import os
import sys
import json

from sohoo.provider import *
from shared.settings import CURRENT_DIR
from sohoo.visualize import analyze_weather_data
from datetime import datetime


def saveJson():
    json_data = get_openMateo._last7days("Indonesia")
    with open(os.path.join(CURRENT_DIR, "weather_last7days.json"), "w") as f:
        json.dump(json_data, f, indent=4)

def main():
    saveJson()
    daily_stats, overall_stats = analyze_weather_data('Indonesia')


if __name__ == "__main__":
    main()
