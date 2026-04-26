"""
Weather Agent
==============
Fetches current weather and forecast from Open-Meteo.

WHY OPEN-METEO?
  - Completely free, no API key needed
  - No rate limiting
  - Global coverage
  - Returns current conditions + hourly/daily forecasts
  - Open source

HOW IT WORKS:
  1. User asks "What's the weather?"
  2. Agent sends a request to Open-Meteo with latitude/longitude
  3. Open-Meteo returns current conditions and forecast
  4. Agent formats it into a natural language summary
  5. Dispatcher sends it to Ollama to phrase conversationally

OPEN-METEO API:
  URL: https://api.open-meteo.com/v1/forecast
  Parameters:
    latitude, longitude — location (from config.py)
    current — what current data to include (temperature, wind, etc.)
    daily — what forecast data to include (max/min temp, precipitation)
    timezone — auto-detect from coordinates

WMO WEATHER CODES:
  Open-Meteo uses WMO (World Meteorological Organization) codes
  to describe weather conditions:
    0 = Clear sky
    1-3 = Partly cloudy
    45-48 = Fog
    51-55 = Drizzle
    61-65 = Rain
    71-77 = Snow
    80-82 = Rain showers
    95-99 = Thunderstorm
  
  We convert these codes to human-readable descriptions.
"""

import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents.base import BaseAgent, FunctionDef, AgentResult
from config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, GRAY, RESET, CYAN, GREEN, YELLOW


# Open-Meteo API endpoint
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather code to description mapping
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherAgent(BaseAgent):
    """Fetches weather data from Open-Meteo."""

    name = "weather"
    description = "Fetches current weather and forecast"

    def __init__(self):
        super().__init__()
        self._http = requests.Session()
        # Default location from config (Brampton, ON)
        self.latitude = DEFAULT_LATITUDE
        self.longitude = DEFAULT_LONGITUDE
        # Cache to avoid hitting the API on every request
        self._cache = None
        self._cache_time = None

    def get_functions(self) -> List[FunctionDef]:
        return [
            FunctionDef(
                name="get_weather",
                description=(
                    "Get the current weather and forecast. Use when the user says: "
                    "'What is the weather?', 'How is it outside?', "
                    "'Do I need an umbrella?', 'Temperature?', "
                    "'Will it rain today?', 'Weather forecast', "
                    "'What is it like outside?'"
                ),
                parameters={},
                required_params=[],
            ),
        ]

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def execute(self, func_name: str, params: Dict[str, Any]) -> AgentResult:
        if func_name == "get_weather":
            return self._get_weather()
        return AgentResult(False, f"Unknown function: {func_name}")

    # ─────────────────────────────────────────
    # Fetch Weather
    # ─────────────────────────────────────────

    def _get_weather(self) -> AgentResult:
        """
        Fetch current weather and today's forecast from Open-Meteo.
        Caches for 10 minutes to avoid unnecessary API calls.
        """
        # Check cache (10 minute TTL)
        if self._cache and self._cache_time:
            elapsed = (datetime.now() - self._cache_time).total_seconds()
            if elapsed < 600:
                print(f"{GRAY}[Weather] Using cached data{RESET}")
                return AgentResult(
                    success=True,
                    message=self._cache,
                    data=self._cache,
                )

        print(f"{CYAN}[Weather] Fetching from Open-Meteo...{RESET}")

        try:
            # Build the API request
            # We ask for current conditions and today's daily summary
            response = self._http.get(
                OPEN_METEO_URL,
                params={
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "current": ",".join([
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "weather_code",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ]),
                    "daily": ",".join([
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "precipitation_probability_max",
                        "weather_code",
                        "sunrise",
                        "sunset",
                    ]),
                    "timezone": "auto",
                    "forecast_days": 3,
                },
                timeout=10,
            )

            if response.status_code != 200:
                return AgentResult(False, f"Weather API returned status {response.status_code}.")

            data = response.json()
            summary = self._format_weather(data)

            # Cache it
            self._cache = summary
            self._cache_time = datetime.now()

            print(f"{GREEN}[Weather] ✓ Weather data fetched{RESET}")

            return AgentResult(
                success=True,
                message=summary,
                data=data,
            )

        except Exception as e:
            print(f"{GRAY}[Weather] Error: {e}{RESET}")
            return AgentResult(False, f"Could not fetch weather: {e}")

    # ─────────────────────────────────────────
    # Format Weather Data
    # ─────────────────────────────────────────

    def _format_weather(self, data: dict) -> str:
        """
        Convert raw Open-Meteo JSON into a human-readable summary
        that the LLM can rephrase naturally.
        """
        lines = []

        # ── Current conditions ──
        current = data.get("current", {})
        if current:
            temp = current.get("temperature_2m", "?")
            feels_like = current.get("apparent_temperature", "?")
            humidity = current.get("relative_humidity_2m", "?")
            wind_speed = current.get("wind_speed_10m", "?")
            weather_code = current.get("weather_code", 0)
            condition = WMO_CODES.get(weather_code, "Unknown")

            lines.append("Current weather:")
            lines.append(f"  Condition: {condition}")
            lines.append(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
            lines.append(f"  Humidity: {humidity}%")
            lines.append(f"  Wind: {wind_speed} km/h")

        # ── Daily forecast (today + next 2 days) ──
        daily = data.get("daily", {})
        if daily:
            dates = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precip_probs = daily.get("precipitation_probability_max", [])
            weather_codes = daily.get("weather_code", [])
            sunrises = daily.get("sunrise", [])
            sunsets = daily.get("sunset", [])

            for i in range(min(3, len(dates))):
                try:
                    date_obj = datetime.strptime(dates[i], "%Y-%m-%d")
                    day_name = date_obj.strftime("%A")
                    if i == 0:
                        day_name = "Today"
                    elif i == 1:
                        day_name = "Tomorrow"

                    condition = WMO_CODES.get(weather_codes[i] if i < len(weather_codes) else 0, "Unknown")
                    high = max_temps[i] if i < len(max_temps) else "?"
                    low = min_temps[i] if i < len(min_temps) else "?"
                    rain_chance = precip_probs[i] if i < len(precip_probs) else "?"

                    lines.append(f"\n{day_name} forecast:")
                    lines.append(f"  {condition}, High {high}°C, Low {low}°C")
                    lines.append(f"  Chance of precipitation: {rain_chance}%")

                    # Sunrise/sunset for today only
                    if i == 0:
                        if i < len(sunrises) and sunrises[i]:
                            sunrise = datetime.fromisoformat(sunrises[i]).strftime("%I:%M %p").lstrip("0")
                            lines.append(f"  Sunrise: {sunrise}")
                        if i < len(sunsets) and sunsets[i]:
                            sunset = datetime.fromisoformat(sunsets[i]).strftime("%I:%M %p").lstrip("0")
                            lines.append(f"  Sunset: {sunset}")

                except (IndexError, ValueError):
                    continue

        return "\n".join(lines) if lines else "Weather data unavailable."

    # ─────────────────────────────────────────
    # System Info
    # ─────────────────────────────────────────

    def get_system_info(self) -> Optional[Dict]:
        """Report current weather for the dashboard."""
        if not self._cache:
            return None

        # Return a simplified version for system info
        return {"weather_summary": self._cache.split("\n")[0:4]}

    def shutdown(self):
        self._http.close()