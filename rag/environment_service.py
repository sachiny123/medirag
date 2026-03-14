"""
Environment Service for MediLens AI.

Fetches weather (OpenWeather) and AQI (WAQI) data for patient location.
Data is cached for 10 minutes to avoid repeat API calls.
"""

import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WAQI_API_KEY = os.getenv("WAQI_API_KEY", "")

# In-memory cache: key = "lat_lon", value = (timestamp, data_dict)
_env_cache: dict = {}
CACHE_TTL_SECONDS = 600  # 10 minutes


def _classify_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    else:
        return "Unhealthy"


def _classify_heat_risk(temp_celsius: float) -> str:
    if temp_celsius > 38:
        return "High"
    elif temp_celsius > 34:
        return "Moderate"
    else:
        return "Low"


def _fetch_weather(lat: float, lon: float) -> dict:
    """Fetch temperature, humidity, and weather condition from OpenWeather."""
    if not OPENWEATHER_API_KEY:
        return {}
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {
            "temperature": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "weather_condition": data["weather"][0]["description"].capitalize(),
        }
    except Exception as e:
        print(f"[EnvService] Weather fetch failed: {e}")
        return {}


def _fetch_aqi(lat: float, lon: float) -> dict:
    """Fetch Air Quality Index from WAQI API."""
    if not WAQI_API_KEY:
        return {}
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_API_KEY}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "ok":
            aqi_val = data["data"]["aqi"]
            return {"aqi": aqi_val}
    except Exception as e:
        print(f"[EnvService] AQI fetch failed: {e}")
    return {}


def get_environment_data(lat: float, lon: float) -> Optional[dict]:
    """
    Retrieve environmental health data for a location.
    Returns structured dict or None if APIs are unavailable/unconfigured.

    Cache key format: "28.61_77.20" (2 decimal places).
    Cache TTL: 10 minutes.
    """
    cache_key = f"{round(lat, 2)}_{round(lon, 2)}"
    now = time.time()

    # Serve from cache if fresh
    if cache_key in _env_cache:
        cached_at, cached_data = _env_cache[cache_key]
        if now - cached_at < CACHE_TTL_SECONDS:
            print(f"[EnvService] Cache hit for {cache_key}")
            return cached_data

    # Both APIs require keys — skip silently if not configured
    if not OPENWEATHER_API_KEY and not WAQI_API_KEY:
        print("[EnvService] No API keys configured. Skipping environmental data.")
        return None

    # Fetch in parallel using standard dict merge
    weather = _fetch_weather(lat, lon)
    aqi_data = _fetch_aqi(lat, lon)

    if not weather and not aqi_data:
        return None

    # Build result
    temp = weather.get("temperature", None)
    aqi = aqi_data.get("aqi", None)

    result = {
        "temperature": temp,
        "humidity": weather.get("humidity", None),
        "weather_condition": weather.get("weather_condition", "Unknown"),
        "aqi": aqi,
        "aqi_category": _classify_aqi(aqi) if aqi is not None else "Unknown",
        "heatstroke_risk": _classify_heat_risk(temp) if temp is not None else "Unknown",
    }

    # Store in cache
    _env_cache[cache_key] = (now, result)
    print(f"[EnvService] Fetched & cached env data for {cache_key}: {result}")
    return result


def build_env_context_block(env_data: dict) -> str:
    """Build a Gemini prompt block from env data."""
    if not env_data:
        return ""
    temp = f"{env_data['temperature']}°C" if env_data.get("temperature") is not None else "N/A"
    humidity = f"{env_data['humidity']}%" if env_data.get("humidity") is not None else "N/A"
    aqi = str(env_data.get("aqi", "N/A"))
    aqi_cat = env_data.get("aqi_category", "N/A")
    heat = env_data.get("heatstroke_risk", "N/A")
    weather = env_data.get("weather_condition", "N/A")

    return (
        f"\nENVIRONMENTAL CONDITIONS\n\n"
        f"Temperature: {temp}\n"
        f"Humidity: {humidity}\n"
        f"AQI: {aqi}\n"
        f"AQI Category: {aqi_cat}\n"
        f"Heatstroke Risk: {heat}\n\n"
        f"Consider how these environmental conditions may contribute to or worsen the patient's symptoms.\n"
    )


def summarize_env_risk(env_data: dict) -> str:
    """Return a human-readable summary of environmental health risks for API response."""
    if not env_data:
        return "No environmental data available."

    parts = []
    heat = env_data.get("heatstroke_risk", "Low")
    aqi_cat = env_data.get("aqi_category", "Good")
    temp = env_data.get("temperature")
    aqi = env_data.get("aqi")

    if heat == "High":
        parts.append(f"Extreme heat ({temp}°C) poses a high heatstroke risk — ensure hydration and avoid outdoor exposure.")
    elif heat == "Moderate":
        parts.append(f"Moderate heat ({temp}°C) detected — stay hydrated and limit prolonged outdoor activity.")

    if aqi_cat == "Unhealthy":
        parts.append(f"Air quality is unhealthy (AQI {aqi}) — may worsen respiratory symptoms or skin conditions.")
    elif aqi_cat == "Unhealthy for Sensitive Groups":
        parts.append(f"Air quality is poor for sensitive groups (AQI {aqi}) — those with allergies or respiratory conditions should take precautions.")
    elif aqi_cat == "Moderate":
        parts.append(f"Moderate air pollution (AQI {aqi}) — limit prolonged outdoor activity if symptoms are respiratory.")

    if not parts:
        return f"Environmental conditions appear normal. Temperature: {temp}°C, AQI: {aqi} ({aqi_cat})."

    return " ".join(parts)
