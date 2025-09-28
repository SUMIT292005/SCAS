# mandi_api.py
import requests

# ================= Configuration =================
API_KEY = os.getenv("API_KEY")
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}?format=json&api-key={API_KEY}"


# ================= Helper Functions =================
def fetch_unique(field, filters=None, limit=1000):
    """Fetch unique values for a given field from API."""
    params = {
        "format": "json",
        "limit": limit
    }
    if filters:
        params.update(filters)
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "records" not in data:
        return []
    return sorted(set(record.get(field) for record in data["records"] if record.get(field)))


def get_states():
    """Fetch all unique states."""
    return fetch_unique("state")


def get_districts(state):
    """Fetch districts for a given state."""
    return fetch_unique("district", {"filters[state]": state})


def get_markets(state, district):
    """Fetch markets for a given state & district."""
    return fetch_unique("market", {"filters[state]": state, "filters[district]": district})


def get_commodities(state, district, market):
    """Fetch commodities for a given state, district, and market."""
    return fetch_unique("commodity", {
        "filters[state]": state,
        "filters[district]": district,
        "filters[market]": market
    })


def get_price(state, district, market, commodity):
    """Fetch mandi price for given inputs."""
    params = {
        "filters[state]": state,
        "filters[district]": district,
        "filters[market]": market,
        "filters[commodity]": commodity,
        "limit": 1
    }
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "records" in data and data["records"]:
        record = data["records"][0]
        return {
            "state": record.get("state"),
            "district": record.get("district"),
            "market": record.get("market"),
            "commodity": record.get("commodity"),
            "variety": record.get("variety"),
            "min_price": float(record.get("min_price", 0)),
            "max_price": float(record.get("max_price", 0)),
            "modal_price": float(record.get("modal_price", 0)),
            "arrival_date": record.get("arrival_date")
        }
    return None
