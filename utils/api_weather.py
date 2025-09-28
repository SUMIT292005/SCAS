import requests
import difflib


# Replace with your actual API key
API_KEY = os.getenv("API_KEY1")


DISTRICT_CITY_MAP = {
    "Mumbai": "Mumbai,IN",
    "Pune": "Pune,IN",
    "Nagpur": "Nagpur,IN",
    "Nashik": "Nashik,IN",
    "Aurangabad": "Aurangabad,IN",
    "Solapur": "Solapur,IN",
    "Thane": "Thane,IN",
    "Kolhapur": "Kolhapur,IN",
    "Amravati": "Amravati,IN",
    "Akola": "Akola,IN",
    "Jalgaon": "Jalgaon,IN",
    "Latur": "Latur,IN",
    "Chandrapur": "Chandrapur,IN",
    "Buldhana": "Buldhana,IN",
    "Parbhani": "Parbhani,IN",
    "Nanded": "Nanded,IN",
    "Wardha": "Wardha,IN",
    "Raigad": "Raigad,IN",
    "Ratnagiri": "Ratnagiri,IN",
    "Sangli": "Sangli,IN",
    "Satara": "Satara,IN",
    "Gadchiroli": "Gadchiroli,IN",
    "Hingoli": "Hingoli,IN",
    "Jalna": "Jalna,IN",
    "Nandurbar": "Nandurbar,IN",
    "Palghar": "Palghar,IN",
    "Bhandara": "Bhandara,IN",
    "Gondia": "Gondia,IN"
}

def get_all_districts():
    return sorted(DISTRICT_CITY_MAP.keys())

def get_weather(district):
    if district not in DISTRICT_CITY_MAP:
        closest = difflib.get_close_matches(district, DISTRICT_CITY_MAP.keys(), n=1)
        raise ValueError(f"District '{district}' not found. Did you mean: {closest[0]}?")
    
    city = DISTRICT_CITY_MAP[district]
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY1}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

    # Average temp & humidity for next 24h
    temps = [entry["main"]["temp"] for entry in data["list"][:8]]
    hums = [entry["main"]["humidity"] for entry in data["list"][:8]]
    avg_temp = sum(temps) / len(temps)
    avg_humidity = sum(hums) / len(hums)

    # Total rainfall
    rainfall = sum(entry.get("rain", {}).get("3h", 0.0) for entry in data["list"][:8])

    return {
        "temperature": round(avg_temp, 1),
        "humidity": round(avg_humidity, 1),
        "rainfall": round(rainfall, 1)
    }
