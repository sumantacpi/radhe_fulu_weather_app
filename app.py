"""
AI Weather Report
-----------------
A Streamlit app that takes a city name and shows the CURRENT weather plus a
7-day forecast. It uses the Anthropic (Claude) API with the built-in web_search
tool to fetch live data, then renders it.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import json
from datetime import date

import pandas as pd
import streamlit as st
import anthropic

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"          # any current model that supports web search
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}

st.set_page_config(page_title="AI Weather Report", page_icon="⛅", layout="centered")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_secret(name: str) -> str:
    """Safely read a value from st.secrets (won't crash if no secrets file)."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def build_prompt(city: str) -> str:
    today = date.today().isoformat()
    return f"""You are a weather reporting assistant. Use the web_search tool to find the
CURRENT weather conditions AND the weather forecast for the next 7 days for the city: "{city}".
Today's date is {today}. Search reputable, up-to-date weather sources.

After researching, respond with ONLY a single valid JSON object (no markdown, no code
fences, no commentary) using EXACTLY this structure:

{{
  "found": true,
  "city": "<resolved city name>",
  "country": "<country>",
  "as_of": "<date/time the data is current as of>",
  "current": {{
    "condition": "<short text, e.g. Partly cloudy>",
    "temp_c": <number>, "temp_f": <number>,
    "feels_like_c": <number>, "feels_like_f": <number>,
    "humidity_pct": <number>,
    "wind_kph": <number>, "wind_mph": <number>
  }},
  "forecast": [
    {{
      "date": "YYYY-MM-DD",
      "day_name": "<e.g. Monday>",
      "condition": "<short text>",
      "high_c": <number>, "low_c": <number>,
      "high_f": <number>, "low_f": <number>,
      "chance_of_rain_pct": <number>
    }}
  ],
  "summary": "<2-4 sentence natural-language summary of the week ahead>"
}}

The "forecast" array MUST contain exactly 7 entries, starting with today.
Use numbers (not strings) for all numeric fields. If a value is unknown, use null.
If you cannot identify the city or find data, respond with ONLY:
{{"found": false, "error": "<reason>"}}
Do not include any text outside the JSON object."""


def extract_text(response) -> str:
    """Concatenate all text blocks from an Anthropic response (ignores tool blocks)."""
    parts = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "".join(parts)


def extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerating fences/preamble."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in the model's response.")
    return json.loads(text[start:end + 1])


def fetch_weather(client: "anthropic.Anthropic", city: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": build_prompt(city)}],
    )
    return extract_json(extract_text(response))


def fmt_temp(c, f) -> str:
    parts = []
    if c is not None:
        parts.append(f"{round(c)}°C")
    if f is not None:
        parts.append(f"{round(f)}°F")
    return " / ".join(parts) if parts else "—"


def render_report(data: dict) -> None:
    city = data.get("city", "")
    country = data.get("country")
    st.subheader(f"{city}{', ' + country if country else ''}")
    if data.get("as_of"):
        st.caption(f"Current as of {data['as_of']}")

    # --- Current conditions ---
    cur = data.get("current", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", fmt_temp(cur.get("temp_c"), cur.get("temp_f")))
    c2.metric("Feels Like", fmt_temp(cur.get("feels_like_c"), cur.get("feels_like_f")))
    hum = cur.get("humidity_pct")
    c3.metric("Humidity", f"{hum}%" if hum is not None else "—")
    wind = cur.get("wind_kph")
    c4.metric("Wind", f"{wind} kph" if wind is not None else "—")
    if cur.get("condition"):
        st.info(f"**Right now:** {cur['condition']}")

    # --- 7-day forecast ---
    forecast = data.get("forecast", []) or []
    if forecast:
        st.markdown("### 7-Day Forecast")

        chart_df = pd.DataFrame(
            {
                "High (°C)": [f.get("high_c") for f in forecast],
                "Low (°C)": [f.get("low_c") for f in forecast],
            },
            index=[f.get("day_name") or f.get("date") or str(i) for i, f in enumerate(forecast)],
        )
        st.line_chart(chart_df)

        table = pd.DataFrame(
            [
                {
                    "Day": f.get("day_name") or "",
                    "Date": f.get("date") or "",
                    "Condition": f.get("condition") or "",
                    "High": fmt_temp(f.get("high_c"), f.get("high_f")),
                    "Low": fmt_temp(f.get("low_c"), f.get("low_f")),
                    "Rain": f"{f.get('chance_of_rain_pct')}%"
                    if f.get("chance_of_rain_pct") is not None else "—",
                }
                for f in forecast
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

    # --- Summary ---
    if data.get("summary"):
        st.markdown("### Summary")
        st.write(data["summary"])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("⛅ AI Weather Report")
st.caption("Live weather and a 7-day forecast, powered by Claude + web search.")

with st.sidebar:
    st.header("Settings")
    key_input = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        help="Stored only in this session. Get one at console.anthropic.com.",
    )
    st.markdown("[Get an API key →](https://console.anthropic.com/)")

city = st.text_input("Enter a city name", placeholder="e.g. London, Tokyo, New York")
go = st.button("Get Weather Report", type="primary")

if go:
    api_key = key_input or get_secret("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.error("Please add your Anthropic API key in the sidebar.")
        st.stop()
    if not city.strip():
        st.warning("Please enter a city name.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)
    with st.spinner(f"Researching live weather for “{city.strip()}”…"):
        try:
            data = fetch_weather(client, city.strip())
        except anthropic.APIError as e:
            st.error(f"Anthropic API error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Could not produce a report: {e}")
            st.stop()

    if not data.get("found", False):
        st.error(data.get("error", "Couldn't find weather data for that city."))
        st.stop()

    render_report(data)
