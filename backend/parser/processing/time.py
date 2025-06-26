import os

import requests

DUCKLING_URL = os.getenv("DUCKLING_URL", "http://localhost:8002")


def parse_with_duckling(text):
    response = requests.post(
        f"{DUCKLING_URL}/parse", data={"locale": "ru_RU", "text": text}
    ).json()
    temporal_points = []
    temporal_intervals = []

    for entity in response:
        if entity["dim"] == "time":
            value = entity["value"]
            grain = value.get("grain")
            if value["type"] == "value":
                temporal_points.append({"point": value["value"], "grain": grain})
            elif value["type"] == "interval":
                temporal_intervals.append(
                    {
                        "start": value["from"]["value"],
                        "end": value["to"]["value"],
                        "grain": grain,
                    }
                )

    return temporal_points, temporal_intervals
