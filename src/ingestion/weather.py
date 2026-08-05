"""Weather features for NYC.

Weather is the second-strongest demand driver after time-of-day, so it is worth
the extra dependency.

**Provider split, and why it exists.** The brief specified NOAA
(``api.weather.gov``). NOAA was verified at build time and *cannot* serve the
training window: ``/stations/{id}/observations`` returns an empty feature
collection for dates more than roughly a week old, so a 3-month backfill is
impossible. Open-Meteo's archive API is therefore used for historical backfill
(free, no API key, hourly temperature/precipitation/wind), and NOAA is used for
the live and forecast path in the ingestion DAG, where its short retention is
not a problem.

Both providers are normalised to the same three columns so the shared feature
builder is agnostic to which one produced a row:

    hour_ts (tz-naive, local NYC time) | temperature_c | precipitation_mm | wind_speed_kmh
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd

from src.config import Config, data_dir, load_config
from src.ingestion.http import get_with_retries

logger = logging.getLogger(__name__)

WEATHER_COLUMNS = ["hour_ts", "temperature_c", "precipitation_mm", "wind_speed_kmh"]


def empty_weather_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=WEATHER_COLUMNS)
    return frame.astype(
        {
            "hour_ts": "datetime64[ns]",
            "temperature_c": "float64",
            "precipitation_mm": "float64",
            "wind_speed_kmh": "float64",
        }
    )


class OpenMeteoClient:
    """Historical + forecast hourly weather. No API key required."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self._lat = float(self._config.require("weather.latitude"))
        self._lon = float(self._config.require("weather.longitude"))
        self._timeout_s = int(self._config.get_path("weather.request_timeout_s", 30))
        self._tz = self._config.get_path("project.timezone", "America/New_York")

    def _request(self, url: str, params: dict[str, object]) -> pd.DataFrame:
        payload = get_with_retries(url, timeout_s=self._timeout_s, params=params).json()
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            logger.warning("Open-Meteo returned no hourly data for params=%s", params)
            return empty_weather_frame()
        frame = pd.DataFrame(
            {
                "hour_ts": pd.to_datetime(times),
                "temperature_c": hourly.get("temperature_2m"),
                "precipitation_mm": hourly.get("precipitation"),
                "wind_speed_kmh": hourly.get("wind_speed_10m"),
            }
        )
        for column in ("temperature_c", "precipitation_mm", "wind_speed_kmh"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[WEATHER_COLUMNS]

    def fetch_archive(self, start: date, end: date) -> pd.DataFrame:
        """Hourly observations for a closed date range (local time)."""
        return self._request(
            self._config.require("weather.open_meteo_archive_url"),
            {
                "latitude": self._lat,
                "longitude": self._lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "timezone": self._tz,
            },
        )

    def fetch_forecast(self, past_days: int = 2, forecast_days: int = 2) -> pd.DataFrame:
        """Recent past + near-future hours, used by the live serving path."""
        return self._request(
            self._config.require("weather.open_meteo_forecast_url"),
            {
                "latitude": self._lat,
                "longitude": self._lon,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "timezone": self._tz,
                "past_days": past_days,
                "forecast_days": forecast_days,
            },
        )


class NOAAClient:
    """NOAA api.weather.gov hourly forecast for the NYC grid point.

    Only usable for the live/near-term path: NOAA discards observations older
    than about a week, which is why historical backfill goes through Open-Meteo.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self._timeout_s = int(self._config.get_path("weather.request_timeout_s", 30))
        self._headers = {
            "User-Agent": self._config.require("weather.noaa_user_agent"),
            "Accept": "application/geo+json",
        }
        self._hourly_url: str | None = None

    def _resolve_hourly_url(self) -> str:
        if self._hourly_url:
            return self._hourly_url
        url = self._config.require("weather.noaa_points_url").format(
            lat=self._config.require("weather.latitude"),
            lon=self._config.require("weather.longitude"),
        )
        payload = get_with_retries(url, timeout_s=self._timeout_s, headers=self._headers).json()
        hourly = payload.get("properties", {}).get("forecastHourly")
        if not hourly:
            raise ValueError(f"NOAA points response had no forecastHourly URL: {url}")
        self._hourly_url = hourly
        return hourly

    @staticmethod
    def _to_celsius(value: float | None, unit: str | None) -> float | None:
        if value is None:
            return None
        if unit and unit.upper().endswith("F"):
            return (value - 32.0) * 5.0 / 9.0
        return value

    @staticmethod
    def _speed_to_kmh(raw: str | float | None) -> float | None:
        """NOAA reports wind as strings like ``"10 to 15 mph"``."""
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        parts = [p for p in str(raw).split() if p.replace(".", "", 1).isdigit()]
        if not parts:
            return None
        value = sum(float(p) for p in parts) / len(parts)
        return value * 1.609344 if "mph" in str(raw).lower() else value

    def fetch_hourly_forecast(self) -> pd.DataFrame:
        payload = get_with_retries(
            self._resolve_hourly_url(), timeout_s=self._timeout_s, headers=self._headers
        ).json()
        periods = payload.get("properties", {}).get("periods") or []
        rows = []
        for period in periods:
            try:
                start = pd.to_datetime(period["startTime"]).tz_localize(None)
            except Exception:  # noqa: BLE001 - skip unparseable periods, keep the rest
                continue
            probability = (period.get("probabilityOfPrecipitation") or {}).get("value")
            rows.append(
                {
                    "hour_ts": start,
                    "temperature_c": self._to_celsius(
                        period.get("temperature"), period.get("temperatureUnit")
                    ),
                    # NOAA's hourly forecast exposes precipitation *probability*,
                    # not an amount. Scaling it to a pseudo-millimetre value keeps
                    # the feature schema identical to the archive path; it is an
                    # approximation, flagged here and in the README.
                    "precipitation_mm": (
                        float(probability) / 100.0 if probability is not None else 0.0
                    ),
                    "wind_speed_kmh": self._speed_to_kmh(period.get("windSpeed")),
                }
            )
        if not rows:
            logger.warning("NOAA returned no usable forecast periods")
            return empty_weather_frame()
        return pd.DataFrame(rows)[WEATHER_COLUMNS]


def load_historical_weather(
    start: datetime | date,
    end: datetime | date,
    config: Config | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Hourly weather covering ``[start, end]``, cached to parquet.

    Never raises: if the provider is unreachable the caller gets an empty frame
    and the feature builder fills weather columns with NaN, which LightGBM
    handles natively. A training run should degrade, not die, on a weather outage.
    """
    cfg = config or load_config()
    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    cache_path = data_dir("cache", f"weather_{start_date}_{end_date}.parquet")
    if use_cache and cache_path.exists():
        logger.info("Using cached weather for %s..%s", start_date, end_date)
        return pd.read_parquet(cache_path)

    try:
        frame = OpenMeteoClient(cfg).fetch_archive(start_date, end_date)
    except Exception as exc:  # noqa: BLE001
        logger.error("Historical weather fetch failed (%s); continuing without weather", exc)
        return empty_weather_frame()

    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset="hour_ts").sort_values("hour_ts").reset_index(drop=True)
    frame.to_parquet(cache_path, index=False)
    logger.info("Fetched %d hours of weather (%s..%s)", len(frame), start_date, end_date)
    return frame


def load_live_weather(config: Config | None = None) -> pd.DataFrame:
    """Weather for the serving path: NOAA first, Open-Meteo as fallback.

    The archive API lags real time by a day or two, so the forecast endpoint is
    what the live path needs. NOAA is tried first because it is the
    authoritative US source; Open-Meteo backs it up because NOAA's grid
    endpoints return 500s more often than one would like.
    """
    cfg = config or load_config()
    provider = cfg.get_path("weather.live_provider", "noaa")

    if provider == "noaa":
        try:
            frame = NOAAClient(cfg).fetch_hourly_forecast()
            if not frame.empty:
                return frame
            logger.warning("NOAA returned an empty forecast; falling back to Open-Meteo")
        except Exception as exc:  # noqa: BLE001
            logger.warning("NOAA live weather failed (%s); falling back to Open-Meteo", exc)

    try:
        return OpenMeteoClient(cfg).fetch_forecast()
    except Exception as exc:  # noqa: BLE001
        logger.error("All live weather providers failed (%s); returning empty frame", exc)
        return empty_weather_frame()


def weather_window_for(hourly_index: pd.Series, config: Config | None = None) -> pd.DataFrame:
    """Convenience: fetch weather spanning the range of a timestamp series."""
    if hourly_index.empty:
        return empty_weather_frame()
    start = pd.Timestamp(hourly_index.min()).date() - timedelta(days=1)
    end = pd.Timestamp(hourly_index.max()).date() + timedelta(days=1)
    return load_historical_weather(start, end, config)
