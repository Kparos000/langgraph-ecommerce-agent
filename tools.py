import re
import json
from datetime import date, timedelta
from langchain_core.tools import tool
from config import get_bq_client

@tool
def query_database(sql: str) -> str:
    """Execute SQL query on BigQuery and return results as JSON."""
    client = get_bq_client()
    try:
        df = client.query(sql).to_dataframe()
        return df.to_json(orient="records")
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def validator(sql: str) -> str:
    """Validate SQL query for basic structure and schema compliance."""
    if not isinstance(sql, str) or not sql.strip():
        return "Invalid: Empty SQL."
    s = sql.strip().upper()
    if not s.startswith("SELECT"):
        return "Invalid: Must start with SELECT."
    if "DROP " in s or "DELETE " in s or "TRUNCATE " in s or "ALTER " in s or "UPDATE " in s or "INSERT " in s or "CREATE " in s:
        return "Invalid: Non-SELECT statements are not allowed."
    if "JOIN" in s and " ON " not in s:
        return "Invalid: JOIN missing ON clause."
    if "BIGQUERY-PUBLIC-DATA.THELOOK_ECOMMERCE" not in s:
        return "Invalid: Must reference bigquery-public-data.thelook_ecommerce."
    return "Valid"

@tool
def generate_final_answer(answer: str) -> str:
    """Generate the final answer (placeholder for synthesis)."""
    return answer

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(days=7 * (n - 1))
    return d

def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d

def _us_holidays(year: int):
    # weekday(): Mon=0,...,Sun=6
    new_year = date(year, 1, 1)
    independence = date(year, 7, 4)
    labor = _nth_weekday_of_month(year, 9, 0, 1)  # first Monday Sep
    halloween = date(year, 10, 31)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)  # 4th Thu Nov (Thu=3)
    black_friday = thanksgiving + timedelta(days=1)
    christmas = date(year, 12, 25)
    memorial = _last_weekday_of_month(year, 5, 0)  # last Monday May
    # Generic "holiday season" retail period (Black Friday through Dec 31)
    holiday_season_start = black_friday
    holiday_season_end = date(year, 12, 31)
    return [
        {"name": "New Year’s Day", "start": str(new_year), "end": str(new_year)},
        {"name": "Memorial Day", "start": str(memorial), "end": str(memorial)},
        {"name": "Independence Day", "start": str(independence), "end": str(independence)},
        {"name": "Labor Day", "start": str(labor), "end": str(labor)},
        {"name": "Halloween", "start": str(halloween), "end": str(halloween)},
        {"name": "Thanksgiving", "start": str(thanksgiving), "end": str(thanksgiving)},
        {"name": "Black Friday", "start": str(black_friday), "end": str(black_friday)},
        {"name": "Christmas", "start": str(christmas), "end": str(christmas)},
        {"name": "Holiday Season", "start": str(holiday_season_start), "end": str(holiday_season_end)},
    ]

def _china_holidays(year: int):
    # Simple fixed-date approximations for demo
    golden_week_start = date(year, 10, 1)
    golden_week_end = date(year, 10, 7)
    singles_day = date(year, 11, 11)
    # Chinese New Year varies; we omit exact calculation to avoid deps
    return [
        {"name": "Golden Week", "start": str(golden_week_start), "end": str(golden_week_end)},
        {"name": "Singles’ Day", "start": str(singles_day), "end": str(singles_day)},
    ]

@tool
def get_holidays(country: str, year: int, holiday: str | None = None) -> str:
    """
    Return JSON list of holiday date ranges for a country and year.
    Supported countries: 'United States', 'US', 'China'.
    If `holiday` is provided, return only that period (case-insensitive partial match).
    """
    c = (country or "").strip().lower()
    if c in ("united states", "us", "usa", "u.s.", "u.s"):
        periods = _us_holidays(int(year))
    elif c == "china":
        periods = _china_holidays(int(year))
    else:
        # Fallback: generic retail holiday season (Nov 1 - Dec 31)
        start = date(int(year), 11, 1)
        end = date(int(year), 12, 31)
        periods = [{"name": "Holiday Season", "start": str(start), "end": str(end)}]

    if holiday:
        h = holiday.lower()
        periods = [p for p in periods if h in p["name"].lower()]

    return json.dumps(periods)
