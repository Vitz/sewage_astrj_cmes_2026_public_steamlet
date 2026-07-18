"""
Weather acquisition for exogenous model features.

Uses Open-Meteo (no API key). Builds lagged covariates required by the RF model:
  T_amb with lag 1 d, P_sum with lag 2 d, H_avg with lag 5 d.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
FALLBACK_CSV = ROOT / "weather_daily_api.csv"

DEFAULT_LAT = 52.2297
DEFAULT_LON = 21.0122
DEFAULT_PLACE = "Warsaw, Poland"

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"

BASE_LABELS = {
    "temperatura_srednia_C": "Mean ambient temperature",
    "opad_suma_mm": "Daily precipitation sum",
    "wilgotnosc_srednia_pct": "Mean relative humidity",
}

BASE_UNITS = {
    "temperatura_srednia_C": "°C",
    "opad_suma_mm": "mm",
    "wilgotnosc_srednia_pct": "%",
}


@dataclass
class WeatherContext:
    source: str
    place_label: str
    latitude: float
    longitude: float
    reference_date: date
    daily: pd.DataFrame
    features: dict[str, float]
    details: dict[str, dict]

    def features_table(self) -> pd.DataFrame:
        from ga_core import FEATURE_META, format_value

        rows = []
        for key, meta in self.details.items():
            fm = FEATURE_META[key]
            rows.append(
                {
                    "Model feature": fm["label"],
                    "Symbol": fm["symbol"],
                    "Value": format_value(key, self.features[key]),
                    "Raw value": self.features[key],
                    "Unit": fm["unit"],
                    "Source day": meta["source_date"],
                    "Lag": f"{meta['lag']} d",
                    "Base variable": BASE_LABELS.get(meta["base"], meta["base"]),
                }
            )
        return pd.DataFrame(rows)


def _http_get_json(url: str, timeout: int = 25) -> dict:
    req = Request(url, headers={"User-Agent": "biogas-ga-optimizer/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode_city(name: str, country_code: str = "PL") -> tuple[float, float, str]:
    params = urlencode(
        {
            "name": name,
            "count": 1,
            "language": "en",
            "format": "json",
            "countryCode": country_code,
        }
    )
    data = _http_get_json(f"{OPEN_METEO_GEOCODE}?{params}")
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Location not found: {name}")
    r = results[0]
    label = f"{r.get('name', name)}"
    if r.get("admin1"):
        label += f", {r['admin1']}"
    if r.get("country"):
        label += f", {r['country']}"
    return float(r["latitude"]), float(r["longitude"]), label


def fetch_open_meteo_daily(
    latitude: float,
    longitude: float,
    past_days: int = 10,
    forecast_days: int = 1,
) -> pd.DataFrame:
    params = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
            "timezone": "Europe/Warsaw",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
    )
    data = _http_get_json(f"{OPEN_METEO_FORECAST}?{params}")
    daily = data["daily"]
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily["time"]),
            "temperatura_srednia_C": daily["temperature_2m_mean"],
            "opad_suma_mm": daily["precipitation_sum"],
            "wilgotnosc_srednia_pct": daily["relative_humidity_2m_mean"],
        }
    )
    return df.sort_values("Date").reset_index(drop=True)


def load_fallback_csv(path: Path = FALLBACK_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fallback weather file missing: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"Data": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_lag_features(
    daily: pd.DataFrame,
    reference: date | None = None,
) -> tuple[dict[str, float], dict[str, dict], date]:
    df = daily.copy()
    date_col = "Date" if "Date" in df.columns else "Data"
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.dropna(
        subset=["temperatura_srednia_C", "opad_suma_mm", "wilgotnosc_srednia_pct"]
    )
    if df.empty:
        raise ValueError("No complete daily weather records available.")

    if reference is None:
        reference = df[date_col].max().date()

    ref_ts = pd.Timestamp(reference)
    needed = {
        "temperatura_srednia_C_lag_1": ("temperatura_srednia_C", 1),
        "opad_suma_mm_lag_2": ("opad_suma_mm", 2),
        "wilgotnosc_srednia_pct_lag_5": ("wilgotnosc_srednia_pct", 5),
    }

    features: dict[str, float] = {}
    details: dict[str, dict] = {}
    indexed = df.set_index(date_col)

    for feat, (base, lag) in needed.items():
        src_day = ref_ts - timedelta(days=lag)
        if src_day not in indexed.index:
            earlier = indexed.index[indexed.index <= src_day]
            if len(earlier) == 0:
                raise ValueError(
                    f"Missing data for {feat} (looked for {src_day.date()}, "
                    f"series starts {df[date_col].min().date()})."
                )
            src_day = earlier.max()
        val = float(indexed.loc[src_day, base])
        features[feat] = val
        details[feat] = {
            "base": base,
            "lag": lag,
            "source_date": str(pd.Timestamp(src_day).date()),
        }

    return features, details, reference


def get_weather_for_optimization(
    *,
    mode: str = "api",
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    place_label: str = DEFAULT_PLACE,
    city: str | None = None,
    reference_date: date | None = None,
    manual_features: dict[str, float] | None = None,
) -> WeatherContext:
    """
    mode:
      - 'api_today' — Open-Meteo for today with study-selected lags (recommended)
      - 'api' — live Open-Meteo (custom location / day)
      - 'csv' — historical plant weather CSV used in the modelling pipeline
      - 'scenario' — fixed meteorological scenario (15 °C / 0 mm / 60 %)
      - 'manual' — user-entered lag features (T_amb−1, P_sum−2, H_avg−5)
    """
    from ga_core import SCENARIO_WEATHER

    if mode == "manual":
        if not manual_features:
            raise ValueError("manual_features required for mode='manual'")
        required = (
            "temperatura_srednia_C_lag_1",
            "opad_suma_mm_lag_2",
            "wilgotnosc_srednia_pct_lag_5",
        )
        features = {k: float(manual_features[k]) for k in required}
        ref = reference_date or date.today()
        details = {
            "temperatura_srednia_C_lag_1": {
                "base": "temperatura_srednia_C",
                "lag": 1,
                "source_date": "manual entry",
            },
            "opad_suma_mm_lag_2": {
                "base": "opad_suma_mm",
                "lag": 2,
                "source_date": "manual entry",
            },
            "wilgotnosc_srednia_pct_lag_5": {
                "base": "wilgotnosc_srednia_pct",
                "lag": 5,
                "source_date": "manual entry",
            },
        }
        return WeatherContext(
            source="Manual entry",
            place_label="user-specified lag values",
            latitude=float("nan"),
            longitude=float("nan"),
            reference_date=ref,
            daily=pd.DataFrame(),
            features=features,
            details=details,
        )

    if mode == "scenario":
        ref = reference_date or date.today()
        features = dict(SCENARIO_WEATHER)
        details = {
            "temperatura_srednia_C_lag_1": {
                "base": "temperatura_srednia_C",
                "lag": 1,
                "source_date": "fixed scenario",
            },
            "opad_suma_mm_lag_2": {
                "base": "opad_suma_mm",
                "lag": 2,
                "source_date": "fixed scenario",
            },
            "wilgotnosc_srednia_pct_lag_5": {
                "base": "wilgotnosc_srednia_pct",
                "lag": 5,
                "source_date": "fixed scenario",
            },
        }
        return WeatherContext(
            source="Fixed meteorological scenario",
            place_label="n/a (scenario values)",
            latitude=float("nan"),
            longitude=float("nan"),
            reference_date=ref,
            daily=pd.DataFrame(),
            features=features,
            details=details,
        )

    if mode == "csv":
        daily = load_fallback_csv()
        features, details, ref = compute_lag_features(daily, reference_date)
        return WeatherContext(
            source=f"Historical CSV ({FALLBACK_CSV.name})",
            place_label="Plant site (pipeline weather archive)",
            latitude=float("nan"),
            longitude=float("nan"),
            reference_date=ref,
            daily=daily,
            features=features,
            details=details,
        )

    is_today = mode == "api_today"
    if is_today:
        reference_date = date.today()

    lat, lon, label = latitude, longitude, place_label
    if city:
        lat, lon, label = geocode_city(city)

    try:
        daily = fetch_open_meteo_daily(lat, lon, past_days=14, forecast_days=1)
        features, details, ref = compute_lag_features(daily, reference_date)
        if is_today:
            source = (
                "Open-Meteo API — optimal for today (recommended). "
                "Uses study-selected lags: T_amb−1 d, P_sum−2 d, H_avg−5 d."
            )
        else:
            source = "Open-Meteo Forecast API"
        return WeatherContext(
            source=source,
            place_label=label,
            latitude=lat,
            longitude=lon,
            reference_date=ref,
            daily=daily,
            features=features,
            details=details,
        )
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError) as exc:
        daily = load_fallback_csv()
        features, details, ref = compute_lag_features(daily, reference_date)
        prefix = (
            "Optimal for today (recommended) — fallback CSV"
            if is_today
            else f"Fallback CSV ({FALLBACK_CSV.name})"
        )
        return WeatherContext(
            source=f"{prefix} — API unavailable: {exc}",
            place_label="Plant site (pipeline weather archive)",
            latitude=lat,
            longitude=lon,
            reference_date=ref,
            daily=daily,
            features=features,
            details=details,
        )
