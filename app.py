"""Macro Quant Desk — English cross-asset research terminal."""
from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen
import os
import hmac
import hashlib
import html
import json
import re
import zipfile
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import requests
import streamlit as st
import yfinance as yf
from ui_theme import inject_theme, apply_quant_theme
from dual_rrg import FLOW_TICKERS, build_flow_rrg, flow_quadrant, convergence_label
from sector_options import sector_option_signal
from tactical_signals import participation_metrics, early_rrg_signals
from trend_attribution import sma_slope_panel, slope_statistics, slope_regime, beta_return_decomposition
from technical_indicators import (TIMEFRAME_SOURCE, resample_ohlcv, bollinger_bands,
    anchored_vwap, adx_di, ssl_hybrid, heikin_trend, candle_patterns)

try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    def argrelextrema(data: np.ndarray, comparator, order: int = 1) -> tuple[np.ndarray]:
        """Safe fallback; SciPy's implementation is used when the declared dependency is installed."""
        values=np.asarray(data); matches=[]
        for index in range(order,len(values)-order):
            center=values[index]
            if all(bool(comparator(center,values[index-offset])) and bool(comparator(center,values[index+offset]))
                   for offset in range(1,order+1)):
                matches.append(index)
        return (np.asarray(matches,dtype=int),)

st.set_page_config(page_title="Macro Quant Desk", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")


def configured_access_key() -> str:
    """Enable the gate only when the deployer explicitly configures a secret."""
    try:
        return str(st.secrets.get("AUTH_PASSWORD",os.getenv("AUTH_PASSWORD",""))).strip()
    except Exception:
        return os.getenv("AUTH_PASSWORD","").strip()


def verify_access_key() -> None:
    candidate=str(st.session_state.get("_portal_key_input", ""))
    if hmac.compare_digest(candidate.encode("utf-8"),configured_access_key().encode("utf-8")):
        st.session_state["_portal_authenticated"]=True
        st.session_state["_portal_key_invalid"]=False
        st.session_state.pop("_portal_key_input",None)
    elif candidate:
        st.session_state["_portal_key_invalid"]=True


_configured_password=configured_access_key()
if _configured_password and not st.session_state.get("_portal_authenticated",False):
    st.markdown("""
    <style>
    .stApp{background:radial-gradient(circle at 50% 0%,#14243b 0,#080d15 48%,#05080e 100%)}
    .block-container{max-width:430px;padding-top:18vh}
    [data-testid="stHeader"]{background:transparent}
    .gate-mark{color:#3ed8ff;font:700 .68rem ui-monospace,monospace;letter-spacing:.20em;text-align:center}
    .gate-title{color:#eef5ff;font:650 1.5rem Inter,sans-serif;letter-spacing:.08em;text-align:center;margin:.55rem 0 .2rem}
    .gate-note{color:#71849f;font:.76rem Inter,sans-serif;text-align:center;margin-bottom:1.35rem}
    </style>
    <div class="gate-mark">PRIVATE RESEARCH ACCESS</div>
    <div class="gate-title">MACRO QUANT DESK</div>
    <div class="gate-note">Enter the configured access key to continue.</div>
    """,unsafe_allow_html=True)
    st.text_input("Access key",type="password",key="_portal_key_input",
                  placeholder="••••••••",on_change=verify_access_key)
    if st.session_state.get("_portal_key_invalid",False):
        st.error("The access key is incorrect.")
    # This executes before any cache function is called or any market request is made.
    st.stop()

SECTORS = {
    "XLK": ("Information Technology", "Technology", 33.0), "XLC": ("Communication Services", "Communications", 9.8),
    "XLY": ("Consumer Discretionary", "Discretionary", 10.4), "XLI": ("Industrials", "Industrials", 8.6),
    "XLE": ("Energy", "Energy", 2.9), "XLF": ("Financials", "Financials", 13.9),
    "XLV": ("Health Care", "Health Care", 9.6), "XLP": ("Consumer Staples", "Staples", 5.2),
    "XLU": ("Utilities", "Utilities", 2.4), "XLRE": ("Real Estate", "Real Estate", 2.0),
    "XLB": ("Materials", "Materials", 2.2),
}
RRG_EXTRA = {"SMH": "Semiconductors", "SOXX": "Semiconductors", "GLD": "Gold"}
COMMODITIES = {"GLD": "Gold", "SLV": "Silver", "USO": "Crude Oil", "CPER": "Copper", "UUP": "U.S. Dollar", "TLT": "20Y+ Treasuries"}

# Representative S&P weights for heat-map area only; all price/return data are live.
HEAT_UNIVERSE = {
    "AAPL": ("Apple", "XLK", 7.0), "MSFT": ("Microsoft", "XLK", 6.2), "NVDA": ("NVIDIA", "XLK", 7.4),
    "AVGO": ("Broadcom", "XLK", 2.2), "ORCL": ("Oracle", "XLK", .9), "CRM": ("Salesforce", "XLK", .6),
    "META": ("Meta", "XLC", 2.6), "GOOGL": ("Alphabet", "XLC", 2.0), "GOOG": ("Alphabet C", "XLC", 1.7),
    "NFLX": ("Netflix", "XLC", .8), "AMZN": ("Amazon", "XLY", 3.8), "TSLA": ("Tesla", "XLY", 1.8),
    "HD": ("Home Depot", "XLY", .7), "MCD": ("McDonald's", "XLY", .5), "GE": ("GE Aerospace", "XLI", .6),
    "RTX": ("RTX", "XLI", .5), "CAT": ("Caterpillar", "XLI", .4), "UNP": ("Union Pacific", "XLI", .3),
    "XOM": ("Exxon Mobil", "XLE", 1.1), "CVX": ("Chevron", "XLE", .6), "COP": ("ConocoPhillips", "XLE", .2),
    "SLB": ("SLB", "XLE", .1), "BRK-B": ("Berkshire", "XLF", 1.6), "JPM": ("JPMorgan", "XLF", 1.5),
    "V": ("Visa", "XLF", 1.0), "MA": ("Mastercard", "XLF", .8), "BAC": ("Bank of America", "XLF", .6),
    "GS": ("Goldman Sachs", "XLF", .3), "LLY": ("Eli Lilly", "XLV", 1.4), "JNJ": ("Johnson & Johnson", "XLV", 1.0),
    "UNH": ("UnitedHealth", "XLV", .8), "ABBV": ("AbbVie", "XLV", .8), "MRK": ("Merck", "XLV", .5),
    "WMT": ("Walmart", "XLP", .8), "COST": ("Costco", "XLP", .7), "PG": ("Procter & Gamble", "XLP", .6),
    "KO": ("Coca-Cola", "XLP", .4), "PEP": ("PepsiCo", "XLP", .3), "NEE": ("NextEra Energy", "XLU", .3),
    "SO": ("Southern Co", "XLU", .2), "DUK": ("Duke Energy", "XLU", .1), "PLD": ("Prologis", "XLRE", .2),
    "AMT": ("American Tower", "XLRE", .2), "EQIX": ("Equinix", "XLRE", .2), "LIN": ("Linde", "XLB", .4),
    "SHW": ("Sherwin-Williams", "XLB", .2), "FCX": ("Freeport-McMoRan", "XLB", .15), "APD": ("Air Products", "XLB", .1),
}
NASDAQ_UNIVERSE = {
    "NVDA": ("NVIDIA", "Electronic Technology", 8.8), "AAPL": ("Apple", "Electronic Technology", 8.1),
    "AVGO": ("Broadcom", "Electronic Technology", 5.0), "AMD": ("AMD", "Electronic Technology", 1.5),
    "MU": ("Micron", "Electronic Technology", .8), "QCOM": ("Qualcomm", "Electronic Technology", .8),
    "TXN": ("Texas Instruments", "Electronic Technology", .7), "AMAT": ("Applied Materials", "Electronic Technology", .7),
    "LRCX": ("Lam Research", "Electronic Technology", .6), "KLAC": ("KLA", "Electronic Technology", .5),
    "ADI": ("Analog Devices", "Electronic Technology", .5), "CSCO": ("Cisco", "Electronic Technology", 1.0),
    "MSFT": ("Microsoft", "Technology Services", 7.4), "GOOGL": ("Alphabet A", "Technology Services", 3.0),
    "GOOG": ("Alphabet C", "Technology Services", 2.8), "META": ("Meta", "Technology Services", 4.0),
    "NFLX": ("Netflix", "Technology Services", 1.6), "INTU": ("Intuit", "Technology Services", .6),
    "ADBE": ("Adobe", "Technology Services", .6), "PANW": ("Palo Alto", "Technology Services", .6),
    "CRWD": ("CrowdStrike", "Technology Services", .5), "PLTR": ("Palantir", "Technology Services", 1.0),
    "SNPS": ("Synopsys", "Technology Services", .4), "CDNS": ("Cadence", "Technology Services", .4),
    "AMZN": ("Amazon", "Retail Trade", 5.3), "COST": ("Costco", "Retail Trade", 1.2),
    "MELI": ("MercadoLibre", "Retail Trade", .4), "JD": ("JD.com", "Retail Trade", .2),
    "TSLA": ("Tesla", "Consumer Durables", 3.3), "BKNG": ("Booking", "Consumer Cyclical", .6),
    "ABNB": ("Airbnb", "Consumer Cyclical", .3), "SBUX": ("Starbucks", "Consumer Cyclical", .3),
    "AMGN": ("Amgen", "Health Technology", .8), "GILD": ("Gilead", "Health Technology", .6),
    "ISRG": ("Intuitive Surgical", "Health Technology", .7), "REGN": ("Regeneron", "Health Technology", .4),
    "AZN": ("AstraZeneca", "Health Technology", .5), "TMUS": ("T-Mobile", "Communications", 1.2),
    "CMCSA": ("Comcast", "Communications", .4), "PEP": ("PepsiCo", "Consumer Non-Durables", .8),
    "MDLZ": ("Mondelez", "Consumer Non-Durables", .3), "HON": ("Honeywell", "Producer Manufacturing", .6),
    "LIN": ("Linde", "Producer Manufacturing", .7), "CSX": ("CSX", "Transportation", .3),
    "MAR": ("Marriott", "Consumer Services", .3), "ADP": ("ADP", "Commercial Services", .5),
    "PYPL": ("PayPal", "Commercial Services", .3), "CEG": ("Constellation Energy", "Utilities", .5),
}
YIELD_TICKERS = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
PERIOD_BARS = {"1M":21,"3M":63,"6M":126,"YTD":None,"1Y":252,"3Y":756,"5Y":1260,"10Y":2520,"MAX":None}
REGIME_PAIRS = {"QQQ / TQQQ":("QQQ","TQQQ"),"SPY / SPXL":("SPY","SPXL"),
                "SOXX / SOXL":("SOXX","SOXL"),"XLK / TECL":("XLK","TECL")}
AI_THEMES = {
    "Macro Sectors & Semiconductors": ["XLK","XLE","XLI","XLY","XLC","XLF","XLV","XLP","XLU","XLRE","XLB","SMH","SOXX"],
    "AI Compute & Semiconductor Leaders": ["NVDA","AMD","TSM","AVGO","ASML","MU"],
    "AI Physical Infrastructure": ["VST","CEG","CCJ","OKLO","VRT","ETN"],
    "Critical Resources & Defense": ["FCX","CPER","LMT","RTX","ITA","XLE"],
}
TACTICAL_NAMES = {
    "NVDA":"NVIDIA","AMD":"Advanced Micro Devices","TSM":"TSMC","AVGO":"Broadcom","ASML":"ASML Holding","MU":"Micron",
    "VST":"Vistra","CEG":"Constellation Energy","CCJ":"Cameco","OKLO":"Oklo","VRT":"Vertiv","ETN":"Eaton",
    "FCX":"Freeport-McMoRan","CPER":"Copper ETF","LMT":"Lockheed Martin","RTX":"RTX","ITA":"Aerospace & Defense ETF",
    "SMH":"Semiconductor ETF",
}
OPTION_POOL = ("QQQ","SPY","SMH","SOXX","NVDA","TSLA","MU","ITA","XLE","XLF","XLK")
OPTION_LABELS = {"SMH":"SMH (VanEck Semiconductor ETF)","SOXX":"SOXX (iShares Semiconductor ETF)",
                 "XLE":"XLE (S&P Energy ETF)","XLF":"XLF (S&P Financials ETF)","XLK":"XLK (S&P Technology ETF)"}
LEVERAGED_ETFS = ("TQQQ","SOXL")
PRIMARY_FLOW_ETFS = {
    "TQQQ": {"label":"TQQQ (3x Nasdaq-100)","threshold":500_000_000.0,"class":"Leveraged ETF"},
    "SOXL": {"label":"SOXL (3x Semiconductors)","threshold":500_000_000.0,"class":"Leveraged ETF"},
    "QQQ": {"label":"QQQ (Nasdaq-100 Core)","threshold":1_500_000_000.0,"class":"Core ETF"},
    "SPY": {"label":"SPY (S&P 500 Core)","threshold":1_500_000_000.0,"class":"Core ETF"},
    "SOXX": {"label":"SOXX (iShares Semiconductors)","threshold":300_000_000.0,"class":"Semiconductor ETF"},
    "SMH": {"label":"SMH (VanEck Semiconductors)","threshold":300_000_000.0,"class":"Semiconductor ETF"},
}
COLORS = {
    "XLK": "#40c4ff", "XLC": "#8b7cff", "XLY": "#ff8f5a", "XLI": "#b7c5d8", "XLE": "#00d7a3",
    "XLF": "#3a86ff", "XLV": "#ff4d76", "XLP": "#e6c85c", "XLU": "#5c7cfa", "XLRE": "#b985ff",
    "XLB": "#8bcf74", "SMH": "#ff3e68", "SOXX": "#f472b6", "GLD": "#f3c94f", "SLV": "#c6d0dc", "USO": "#f26d4b",
    "CPER": "#d99152", "UUP": "#38bdf8", "TLT": "#a78bfa", "QQQ": "#ff8a3d",
}
with st.sidebar:
    theme_choice = st.selectbox(
        "Terminal theme",
        ["Bloomberg Pro", "Cyberpunk 2077", "Matrix Green"],
        index=0,
        key="portal_theme_choice",
    )
inject_theme(st.session_state.portal_theme_choice)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Bulk-download adjusted closes and normalize yfinance output shapes."""
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=True, group_by="column")
    if raw.empty:
        return pd.DataFrame(columns=list(tickers))
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        field = "Close" if "Close" in fields else "Adj Close"
        close = raw[field].copy()
    else:
        field = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[field]].copy()
        close.columns = [tickers[0]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.reindex(columns=list(tickers)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).sort_index()


@st.cache_data(ttl=45,show_spinner=False)
def fetch_live_pulse() -> pd.DataFrame:
    """Lightweight intraday tape isolated from the 15-minute research cache."""
    symbols=("SPY","QQQ","IWM","^VIX"); eastern=pytz.timezone("US/Eastern")

    def close_panel(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty: return pd.DataFrame(columns=symbols)
        if isinstance(raw.columns,pd.MultiIndex):
            level0=raw.columns.get_level_values(0)
            panel=raw["Close"].copy() if "Close" in level0 else raw.xs("Close",axis=1,level=1).copy()
        else:
            panel=raw[["Close"]].copy(); panel.columns=[symbols[0]]
        if isinstance(panel,pd.Series): panel=panel.to_frame()
        return panel.reindex(columns=symbols).apply(pd.to_numeric,errors="coerce")

    try:
        minute_raw=yf.download(list(symbols),period="5d",interval="1m",prepost=True,auto_adjust=False,
                               progress=False,threads=True,group_by="column")
    except Exception:
        minute_raw=pd.DataFrame()
    try:
        daily_raw=yf.download(list(symbols),period="10d",interval="1d",auto_adjust=False,
                              progress=False,threads=True,group_by="column")
    except Exception:
        daily_raw=pd.DataFrame()
    minute,daily=close_panel(minute_raw),close_panel(daily_raw)
    rows=[]
    for symbol in symbols:
        intraday=minute[symbol].dropna() if symbol in minute else pd.Series(dtype=float)
        day=daily[symbol].dropna() if symbol in daily else pd.Series(dtype=float)
        latest=np.nan; previous=np.nan; stamp=pd.NaT
        if not intraday.empty:
            latest=float(intraday.iloc[-1]); stamp=pd.Timestamp(intraday.index[-1])
            if stamp.tzinfo is None: stamp=eastern.localize(stamp.to_pydatetime())
            else: stamp=stamp.tz_convert(eastern)
            day_dates=[pd.Timestamp(index).date() for index in day.index]
            if day_dates:
                previous=float(day.iloc[-2] if day_dates[-1]>=stamp.date() and len(day)>=2 else day.iloc[-1])
        elif not day.empty:
            latest=float(day.iloc[-1]); previous=float(day.iloc[-2]) if len(day)>=2 else np.nan
            stamp=pd.Timestamp(day.index[-1])
            stamp=eastern.localize(datetime.combine(stamp.date(),dt_time(16,0)))
        change=(latest/previous-1)*100 if np.isfinite(latest) and np.isfinite(previous) and previous!=0 else np.nan
        rows.append({"Ticker":"VIX" if symbol=="^VIX" else symbol,"YahooTicker":symbol,"Price":latest,
                     "ChangePct":change,"QuoteTimeET":stamp})
    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(ttl=900, show_spinner=False)
def load_custom_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    """Fetch arbitrary user-entered tickers without requiring a local data file."""
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True, group_by="column")
    if raw.empty:
        return pd.DataFrame(columns=list(tickers))
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        field = "Close" if "Close" in fields else "Adj Close"
        close = raw[field].copy()
    else:
        field = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[field]].copy()
        close.columns = [tickers[0]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.reindex(columns=list(tickers)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def cached_sma_slope_panel(close: pd.Series) -> pd.DataFrame:
    return sma_slope_panel(close)


@st.cache_data(ttl=900, show_spinner=False)
def cached_beta_decomposition(target: pd.Series, market: pd.Series) -> dict[str, float] | None:
    return beta_return_decomposition(target, market)


@st.cache_data(ttl=900, show_spinner=False)
def load_sp500_companies() -> tuple[str, ...]:
    """Load a 500-company breadth universe (secondary duplicate share classes removed)."""
    # CSV-only sources keep Community Cloud deployment independent of optional lxml/html parsers.
    urls = ["https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
            "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"]
    table = pd.DataFrame()
    for url in urls:
        try:
            request = Request(url,headers={"User-Agent":"Mozilla/5.0 QuantDashboard/1.0"})
            payload = urlopen(request,timeout=20).read().decode("utf-8")
            table = pd.read_csv(StringIO(payload))
            if "Symbol" in table and len(table) >= 500:
                break
        except Exception:
            continue
    if table.empty or "Symbol" not in table:
        raise ValueError("S&P 500 constituent list unavailable")
    symbols = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    # The index has 503 listed securities but 500 companies; use one class for these three duplicates.
    symbols = [s for s in symbols if s not in {"GOOG", "FOX", "NWS"}]
    return tuple(symbols[:500])


@st.cache_data(ttl=900,show_spinner=False)
def load_sp500_breadth_prices(symbols: tuple[str,...]) -> pd.DataFrame:
    """Chunk the breadth download to reduce partial failures and Yahoo throttling."""
    parts=[]
    for start in range(0,len(symbols),80):
        chunk=symbols[start:start+80]
        try:
            part=load_custom_prices(tuple(chunk),"5d")
            if not part.empty: parts.append(part)
        except Exception:
            continue
    return pd.concat(parts,axis=1).reindex(columns=list(symbols)) if parts else pd.DataFrame(columns=list(symbols))


@st.cache_data(ttl=900, show_spinner=False)
def load_flow_data(tickers: tuple[str, ...], period: str = "3mo") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted closes and volume for signed-dollar-volume flow proxies."""
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True, group_by="column")
    if raw.empty:
        return pd.DataFrame(columns=list(tickers)), pd.DataFrame(columns=list(tickers))
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy() if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
        volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        close = raw[["Close"]].copy(); close.columns = [tickers[0]]
        volume = raw[["Volume"]].copy(); volume.columns = [tickers[0]]
    for frame in (close, volume):
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return (close.reindex(columns=list(tickers)).apply(pd.to_numeric,errors="coerce"),
            volume.reindex(columns=list(tickers)).apply(pd.to_numeric,errors="coerce"))


@st.cache_data(ttl=900,show_spinner=False)
def load_ohlcv(tickers: tuple[str,...], period: str = "max") -> pd.DataFrame:
    """Batch-download full adjusted OHLCV history and return ticker-first columns."""
    raw=yf.download(list(tickers),period=period,auto_adjust=True,progress=False,threads=True,group_by="ticker")
    fields=["Open","High","Low","Close","Volume"]
    frames={}
    if raw.empty:
        return pd.DataFrame()
    for ticker in tickers:
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                top=raw.columns.get_level_values(0)
                frame=raw[ticker].copy() if ticker in top else raw.xs(ticker,axis=1,level=1).copy()
            else:
                frame=raw.copy()
            frame=frame.reindex(columns=fields).apply(pd.to_numeric,errors="coerce").dropna(subset=["Open","High","Low","Close"])
            frame.index=pd.to_datetime(frame.index).tz_localize(None)
            if not frame.empty: frames[ticker]=frame
        except Exception:
            continue
    return pd.concat(frames,axis=1) if frames else pd.DataFrame()


@st.cache_data(ttl=900,show_spinner=False)
def load_technical_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch native Yahoo bars and aggregate only where Yahoo has no native interval."""
    interval,period=TIMEFRAME_SOURCE[timeframe]
    history_args={"interval":interval,"auto_adjust":True,"prepost":False}
    if period.endswith("d"):
        # Yahoo rejects arbitrary range strings on some symbols; explicit dates
        # also keep intraday requests inside its rolling retention window.
        now_et=pd.Timestamp.now(tz="America/New_York")
        history_args["start"]=now_et-pd.Timedelta(days=int(period[:-1])-1)
        history_args["end"]=now_et+pd.Timedelta(days=1)
    else:
        history_args["period"]=period
    raw=yf.Ticker(symbol).history(**history_args)
    if raw.empty:
        return pd.DataFrame()
    fields=["Open","High","Low","Close","Volume"]
    data=raw.reindex(columns=fields).apply(pd.to_numeric,errors="coerce")
    data=data.dropna(subset=fields[:4])
    if data.index.tz is not None:
        data.index=data.index.tz_convert("America/New_York").tz_localize(None)
    else:
        data.index=pd.to_datetime(data.index)
    data=data[~data.index.duplicated(keep="last")].sort_index()
    return resample_ohlcv(data,timeframe)


@st.cache_data(ttl=900, show_spinner=False)
def load_market_news(symbols: tuple[str, ...], per_symbol: int = 6) -> pd.DataFrame:
    """Normalize both legacy and current yfinance news payloads."""
    rows = []
    for symbol in symbols:
        try:
            items = yf.Ticker(symbol).news or []
        except Exception:
            continue
        for item in items[:per_symbol]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            provider = content.get("provider", {})
            source = provider.get("displayName") if isinstance(provider, dict) else provider
            source = source or item.get("publisher") or symbol
            url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
            url = url or item.get("link")
            published = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = pd.to_datetime(published, unit="s", utc=True, errors="coerce")
            else:
                published = pd.to_datetime(published, utc=True, errors="coerce")
            if title and url:
                rows.append({"Time":published,"Source":source,"Headline":title,"URL":url,"Query":symbol})
    if not rows:
        return pd.DataFrame(columns=["Time","Source","Headline","URL","Query"])
    result = pd.DataFrame(rows).drop_duplicates("Headline")
    result["Time"] = pd.to_datetime(result["Time"],utc=True,errors="coerce")
    result["Time"] = result["Time"].dt.tz_convert("Asia/Tokyo").dt.strftime("%m-%d %H:%M")
    return result.head(24)


def us_market_status(now_et: datetime | None = None) -> dict[str,str]:
    """US/Eastern session badge and countdown using weekday cash-session conventions."""
    eastern=pytz.timezone("US/Eastern")
    now=now_et or datetime.now(eastern)
    if now.tzinfo is None:
        now=eastern.localize(now)
    today=now.date(); clock=now.time().replace(tzinfo=None); weekday=now.weekday()
    pre_open,regular_open,regular_close,after_close=dt_time(4,0),dt_time(9,30),dt_time(16,0),dt_time(20,0)
    if weekday<5 and pre_open<=clock<regular_open:
        label,color,event="🟡 PRE-MARKET","#f6c453","until open"
        target=eastern.localize(datetime.combine(today,regular_open))
    elif weekday<5 and regular_open<=clock<regular_close:
        label,color,event="🟢 MARKET OPEN","#31d6a0","until close"
        target=eastern.localize(datetime.combine(today,regular_close))
    elif weekday<5 and regular_close<=clock<after_close:
        label,color,event="🔵 AFTER-HOURS","#4f86d9","until next open"
        next_day=today+timedelta(days=1)
        while next_day.weekday()>=5: next_day+=timedelta(days=1)
        target=eastern.localize(datetime.combine(next_day,regular_open))
    else:
        label,color,event="🔴 MARKET CLOSED","#697689","until next open"
        next_day=today if weekday<5 and clock<regular_open else today+timedelta(days=1)
        while next_day.weekday()>=5: next_day+=timedelta(days=1)
        target=eastern.localize(datetime.combine(next_day,regular_open))
    seconds=max(0,int((target-now).total_seconds())); hours,rem=divmod(seconds,3600); minutes=rem//60
    return {"label":label,"color":color,"time":now.strftime("%Y-%m-%d %H:%M ET"),
            "countdown":f"{event} {hours:02d}h {minutes:02d}m"}


@st.cache_data(ttl=900,show_spinner=False)
def load_option_snapshot(symbol: str,max_expiries: int=2,allow_saved_snapshot: bool=False) -> tuple[pd.DataFrame,float]:
    """Download 7-30 DTE option chains; individual expiries fail independently."""
    ticker=yf.Ticker(symbol); spot=np.nan
    try:
        spot=float(ticker.fast_info["last_price"])
    except Exception:
        try:
            history=ticker.history(period="5d",auto_adjust=True)
            spot=float(history["Close"].dropna().iloc[-1])
        except Exception:
            pass
    try:
        expiries=list(ticker.options or [])
    except Exception:
        expiries=[]
    today=datetime.now(pytz.timezone("US/Eastern")).date(); selected=[]
    for expiry in expiries:
        try:
            dte=(pd.Timestamp(expiry).date()-today).days
            if 7<=dte<=30: selected.append(expiry)
        except Exception:
            continue
    frames=[]
    for expiry in selected[:max(1,max_expiries)]:
        try:
            chain=ticker.option_chain(expiry)
            for option_type,source in (("Call",chain.calls),("Put",chain.puts)):
                frame=source.copy()
                if frame.empty: continue
                frame["Type"]=option_type; frame["Expiration"]=expiry
                frames.append(frame)
        except Exception:
            continue
    if not frames:
        if not allow_saved_snapshot:
            return pd.DataFrame(),spot
        try:
            saved=pd.read_csv(Path(__file__).resolve().parent/"data"/"options_chain_latest.csv")
            saved=saved[saved["Symbol"].eq(symbol)].copy()
            saved["SnapshotAtUTC"]=pd.to_datetime(saved["SnapshotAtUTC"],utc=True,errors="coerce")
            newest=saved["SnapshotAtUTC"].max()
            if pd.isna(newest) or pd.Timestamp.now(tz="UTC")-newest>pd.Timedelta(days=7):
                return pd.DataFrame(),spot
            saved=saved[saved["SnapshotAtUTC"].eq(newest)]
            saved["Expiration"]=saved["Expiration"].astype(str)
            saved=saved[pd.to_datetime(saved["Expiration"],errors="coerce").dt.date.ge(today)]
            saved=saved[saved["Expiration"].isin(sorted(saved["Expiration"].unique())[:max(1,max_expiries)])]
            if saved.empty: return pd.DataFrame(),spot
            for column in ("strike","volume","openInterest","impliedVolatility","bid","ask","lastPrice"):
                saved[column]=pd.to_numeric(saved[column],errors="coerce")
            spot=float(pd.to_numeric(saved["SpotAtCapture"],errors="coerce").dropna().iloc[-1])
            saved.attrs["source"]="SAVED SNAPSHOT"
            saved.attrs["asof_utc"]=newest.isoformat()
            return saved,spot
        except (OSError,KeyError,ValueError,IndexError,pd.errors.ParserError):
            return pd.DataFrame(),spot
    result=pd.concat(frames,ignore_index=True)
    for column in ("strike","volume","openInterest","impliedVolatility","bid","ask","lastPrice"):
        if column not in result: result[column]=np.nan
        result[column]=pd.to_numeric(result[column],errors="coerce")
    result.attrs["source"]="LIVE YAHOO"
    result.attrs["asof_utc"]=pd.Timestamp.now(tz="UTC").isoformat()
    return result,spot


def unusual_option_rows(chain: pd.DataFrame,spot: float,ratio_floor: float=1.5,volume_floor: int=1000,
                        moneyness_limit: float=.20) -> pd.DataFrame:
    if chain.empty or not np.isfinite(spot):
        return pd.DataFrame(columns=["Expiration","Strike","Moneyness","Type","Vol","OI","Vol/OI","IV","Signal"])
    data=chain.copy(); data["volume"]=data["volume"].fillna(0); data["openInterest"]=data["openInterest"].fillna(0)
    data["Vol/OI"]=data["volume"].div(data["openInterest"].replace(0,np.nan))
    data["Moneyness"]=data["strike"].sub(spot).div(spot).mul(100)
    otm=((data["Type"]=="Call")&(data["strike"]>spot))|((data["Type"]=="Put")&(data["strike"]<spot))
    near_spot=data["strike"].sub(spot).abs().div(spot)<=moneyness_limit
    data=data[otm&near_spot&(data["volume"]>=volume_floor)&(data["Vol/OI"]>=ratio_floor)].copy()
    if data.empty:
        return pd.DataFrame(columns=["Expiration","Strike","Moneyness","Type","Vol","OI","Vol/OI","IV","Signal"])
    data["Signal"]=np.where(data["Type"].eq("Call"),"🔥 OTM Call Activity","🛡️ OTM Put Activity")
    data["IV"]=data["impliedVolatility"]*100
    data=data.rename(columns={"strike":"Strike","volume":"Vol","openInterest":"OI"})
    return data[["Expiration","Strike","Moneyness","Type","Vol","OI","Vol/OI","IV","Signal"]].sort_values(["Vol/OI","Vol"],ascending=False)


def continuation_price_diagnostics(ohlcv: pd.DataFrame) -> dict:
    """Diagnose whether a recent high-volume breakout is holding through consolidation."""
    empty={"detected":False,"reason":"No RVOL ≥ 2.0 and return ≥ 3% breakout was detected in the last 10 sessions."}
    required={"High","Low","Close","Volume"}
    if ohlcv is None or ohlcv.empty or not required.issubset(ohlcv.columns):
        return empty
    data=ohlcv[list(required)].apply(pd.to_numeric,errors="coerce").sort_index()
    data=data.dropna(subset=["High","Low","Close","Volume"])
    data=data[data["Volume"]>0]
    if len(data)<22:
        return {"detected":False,"reason":"Fewer than 22 valid OHLCV sessions; breakout relative volume cannot be calculated."}

    data["Return"]=data["Close"].pct_change()
    # Exclude the event day from the denominator so RVOL is not diluted by itself.
    data["AvgVolume20"]=data["Volume"].shift(1).rolling(20,min_periods=10).mean()
    data["RVOL"]=data["Volume"].div(data["AvgVolume20"].replace(0,np.nan))
    candidates=data.tail(10)
    candidates=candidates[(candidates["RVOL"]>=2.0)&(candidates["Return"]>=.03)]
    if candidates.empty:
        return empty

    anchor=pd.Timestamp(candidates.index[-1])
    anchor_loc=int(data.index.get_loc(anchor)); anchored=data.loc[anchor:].copy()
    typical=(anchored["High"]+anchored["Low"]+anchored["Close"])/3
    cumulative_volume=anchored["Volume"].cumsum().replace(0,np.nan)
    avwap=typical.mul(anchored["Volume"]).cumsum().div(cumulative_volume)
    latest_price=float(data["Close"].iloc[-1]); latest_avwap=float(avwap.iloc[-1])
    ema20=data["Close"].ewm(span=20,adjust=False,min_periods=20).mean()
    latest_ema20=float(ema20.iloc[-1]) if np.isfinite(ema20.iloc[-1]) else np.nan
    ema_distance=abs(latest_price/latest_ema20-1)*100 if np.isfinite(latest_ema20) and latest_ema20 else np.nan
    breakout_volume=float(data.loc[anchor,"Volume"])
    post_volume=data.loc[anchor:,"Volume"].iloc[1:]
    dryup=float(post_volume.mean()/breakout_volume) if len(post_volume) and breakout_volume else np.nan
    avwap_pass=bool(latest_price>=latest_avwap)
    dryup_pass=bool(np.isfinite(dryup) and dryup<=.45)
    ema_pass=bool(np.isfinite(ema_distance) and ema_distance<=2.5)

    avwap_label="🟢 Above anchored institutional cost" if avwap_pass else "🚨 Below anchored institutional cost"
    if not np.isfinite(dryup):
        dryup_label="⏳ Breakout is recent; awaiting consolidation volume"
    elif dryup<=.45:
        dryup_label="🟢 Constructive volume dry-up"
    elif dryup>.70:
        dryup_label="⚠️ High-volume stall / distribution risk"
    else:
        dryup_label="🟡 Moderate volume contraction"
    if not np.isfinite(ema_distance):
        ema_label="⚪ Insufficient EMA20 history"
    elif ema_distance<=2.5:
        ema_label="🟢 Constructive EMA20 retest"
    elif ema_distance>6:
        ema_label="⏳ Extended above EMA20"
    else:
        ema_label="🟡 Converging toward EMA20"
    return {
        "detected":True,"anchor":anchor,"days_since":len(data)-1-anchor_loc,
        "breakout_return":float(data.loc[anchor,"Return"]*100),"breakout_rvol":float(data.loc[anchor,"RVOL"]),
        "price":latest_price,"avwap":latest_avwap,"avwap_series":avwap,"avwap_pass":avwap_pass,
        "dryup":dryup,"dryup_pass":dryup_pass,"ema20":latest_ema20,"ema_distance":ema_distance,
        "ema_pass":ema_pass,"avwap_label":avwap_label,"dryup_label":dryup_label,"ema_label":ema_label,
    }


def continuation_option_oi_carry(symbol: str,chain: pd.DataFrame,spot: float) -> dict:
    """Compare near-spot unusual Call OI across locally observed ET-date snapshots.

    Yahoo exposes only current OI, not historical OI. This condition is counted only
    after the session has observed the same contracts on at least two different ET dates.
    """
    neutral={"passed":None,"label":"⚪ OI Increase Pending Next Snapshot","detail":"Yahoo provides no historical OI; the first observation is not scored"}
    if chain is None or chain.empty or not np.isfinite(spot):
        return {"passed":None,"label":"⚪ Option Chain Unavailable","detail":"The request returned no 7–30 DTE contracts"}
    data=chain.copy()
    for column in ("strike","volume","openInterest"):
        if column not in data: data[column]=np.nan
        data[column]=pd.to_numeric(data[column],errors="coerce")
    calls=data[(data.get("Type")=="Call")&(data["strike"]>=spot)&(data["strike"]<=spot*1.15)].copy()
    if calls.empty:
        return {"passed":None,"label":"⚪ No Near-Dated Calls","detail":"No trackable contract was found within 15% above spot"}
    calls["volume"]=calls["volume"].fillna(0); calls["openInterest"]=calls["openInterest"].fillna(0)
    calls["Vol/OI"]=calls["volume"].div(calls["openInterest"].replace(0,np.nan))
    unusual=calls[(calls["volume"]>=500)&(calls["Vol/OI"]>=1.5)]

    def contract_key(row: pd.Series) -> str:
        raw=row.get("contractSymbol")
        if pd.notna(raw) and str(raw): return str(raw)
        return f"{row.get('Expiration','')}|{float(row['strike']):.3f}"

    current_map={contract_key(row):float(row["openInterest"]) for _,row in calls.iterrows()}
    watch=[contract_key(row) for _,row in unusual.iterrows()]
    et_day=datetime.now(pytz.timezone("US/Eastern")).date().isoformat()
    history=st.session_state.setdefault("_continuation_oi_history",{})
    symbol_history=history.setdefault(symbol,{})
    symbol_history[et_day]={"oi":current_map,"watch":watch}
    for old_day in sorted(symbol_history)[:-4]:
        symbol_history.pop(old_day,None)
    observed_days=sorted(symbol_history)
    if len(observed_days)<2:
        if watch:
            neutral["detail"]=f"Created a baseline for {len(watch)} active Call contracts; awaiting the next session"
        else:
            neutral["detail"]="No near-spot Call met Vol/OI ≥ 1.5 and volume ≥ 500 today"
        return neutral
    prior=symbol_history[observed_days[-2]]; current=symbol_history[observed_days[-1]]
    tracked=set(prior.get("watch",[])); common=tracked.intersection(current.get("oi",{}))
    if not common:
        return {"passed":None,"label":"⚪ OI Carry Not Comparable","detail":"The prior active contract expired or was not returned"}
    delta=sum(current["oi"][key]-prior["oi"].get(key,0) for key in common)
    if delta>0:
        return {"passed":True,"label":"🟢 Call OI Accumulation Confirmed","detail":f"Same-contract OI increased by {delta:,.0f}"}
    return {"passed":False,"label":"⚠️ Call OI Not Sustained","detail":f"Same-contract OI changed by {delta:,.0f}"}


@st.cache_data(ttl=900,show_spinner=False)
def load_theme_option_uoa(symbols: tuple[str,...]) -> pd.DataFrame:
    """Scan one selected theme; a failure in one ticker/expiry never blocks the table."""
    rows=[]
    for symbol in symbols:
        try:
            chain,spot=load_option_snapshot(symbol,8)
            unusual=unusual_option_rows(chain,spot,ratio_floor=1.5,volume_floor=500,moneyness_limit=.15)
            if unusual.empty: continue
            unusual.insert(0,"Ticker",symbol)
            unusual["Signal"]=np.where(unusual["Type"].eq("Call"),
                "🔥 Near-OTM Call Accumulation Candidate","🛡️ Near-OTM Put Defense Candidate")
            rows.append(unusual)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Ticker","Expiration","Strike","Moneyness","Type","Vol","OI","Vol/OI","Signal"])
    result=pd.concat(rows,ignore_index=True)
    return result[["Ticker","Expiration","Strike","Moneyness","Type","Vol","OI","Vol/OI","Signal"]].sort_values(
        ["Vol/OI","Vol"],ascending=False).reset_index(drop=True)


@st.cache_data(ttl=900,show_spinner=False)
def load_sector_option_sentiment(symbols: tuple[str,...]) -> dict[str,dict[str,object]]:
    """Read the scheduled, dated option snapshot once; never block Tab 01 on 13 live chains."""
    result={symbol:{"label":"N/A · Option Chain Unavailable","call_vol":0,"put_vol":0,"contracts":0} for symbol in symbols}
    path=Path(__file__).resolve().parent/"data"/"options_chain_latest.csv"
    try:
        saved=pd.read_csv(path)
        saved["SnapshotAtUTC"]=pd.to_datetime(saved["SnapshotAtUTC"],utc=True,errors="coerce")
        now=pd.Timestamp.now(tz="UTC")
        today=now.tz_convert("US/Eastern").date()
        for symbol in symbols:
            frame=saved[saved["Symbol"].eq(symbol)].copy()
            if frame.empty: continue
            newest=frame["SnapshotAtUTC"].max()
            if pd.isna(newest) or now-newest>pd.Timedelta(days=7): continue
            frame=frame[frame["SnapshotAtUTC"].eq(newest)]
            spot=pd.to_numeric(frame["SpotAtCapture"],errors="coerce").dropna()
            if spot.empty: continue
            signal=sector_option_signal(frame,float(spot.iloc[-1]),today)
            signal["asof_utc"]=newest.isoformat()
            result[symbol]=signal
    except (OSError,KeyError,ValueError,pd.errors.ParserError):
        pass
    return result


def theme_tactical_scoreboard(price_frame: pd.DataFrame,ohlcv: pd.DataFrame,tickers: list[str],benchmark: str) -> pd.DataFrame:
    """Cross-sectional price, relative-strength regime and relative-volume snapshot."""
    rows=[]
    for ticker in tickers:
        if ticker not in price_frame or benchmark not in price_frame: continue
        pair=price_frame[[ticker,benchmark]].dropna()
        if len(pair)<200: continue
        ratio=pair[ticker].div(pair[benchmark]).replace([np.inf,-np.inf],np.nan).dropna()
        if len(ratio)<200: continue
        latest=float(ratio.iloc[-1]); ema20=float(ratio.ewm(span=20,adjust=False).mean().iloc[-1])
        ema50=float(ratio.ewm(span=50,adjust=False).mean().iloc[-1]); sma200=float(ratio.rolling(200,min_periods=200).mean().iloc[-1])
        d20=(latest/ema20-1)*100 if ema20 else np.nan; d50=(latest/ema50-1)*100 if ema50 else np.nan
        above20,above50,above200=latest>ema20,latest>ema50,latest>sma200
        if above20 and above50 and above200: regime="🔥 Absolute Strength"
        elif above20 and above50: regime="⚡ Counter-Trend Rebound"
        elif above200: regime="⚠️ Strong Pullback"
        else: regime="❄️ Weak Breakdown"
        close=pair[ticker].dropna(); price=float(close.iloc[-1]); day_return=last_return(close,1)
        metrics=participation_metrics(pd.DataFrame())
        try:
            ticker_frame=ohlcv[ticker].dropna(subset=["Close"]) if ticker in ohlcv.columns.get_level_values(0) else pd.DataFrame()
            metrics=participation_metrics(ticker_frame)
        except Exception:
            pass
        rvol=metrics["rvol"]; dollar_volume=metrics["dollar_volume_m"]
        volume_trend=metrics["volume_trend_pct"]; avwap_distance=metrics["avwap_distance_pct"]
        rows.append({
            "Ticker":f"{ticker} · {TACTICAL_NAMES.get(ticker,SECTORS.get(ticker,(ticker,ticker,0))[1])}",
            "Price":f"${price:,.2f} · {day_return:+.2f}%" if np.isfinite(day_return) else f"${price:,.2f}",
            "RS vs 20EMA":f"{'🟢 Above' if above20 else '🔴 Below'} ({d20:+.2f}%)",
            "RS vs 50EMA":f"{'🟢 Above' if above50 else '🔴 Below'} ({d50:+.2f}%)",
            "RVOL (20D)":f"🔥 Volume Surge · {rvol:.2f}x" if np.isfinite(rvol) and rvol>=2.0 else (f"{rvol:.2f}x" if np.isfinite(rvol) else "N/A"),
            "Dollar Volume ($M)":f"${dollar_volume:,.1f}" if np.isfinite(dollar_volume) else "N/A",
            "Volume Trend (5D vs 20D)":f"{volume_trend:+.1f}%" if np.isfinite(volume_trend) else "N/A",
            "AVWAP Cost Basis":(f"🟢 Above ({avwap_distance:+.1f}%)" if avwap_distance>=0 else
                                    f"🔴 Below ({avwap_distance:+.1f}%)") if np.isfinite(avwap_distance) else "N/A",
            "Regime":regime,
            "_MomentumScore":float(np.nan_to_num(d20)+np.nan_to_num(d50)+np.nan_to_num(day_return)*.25),
            "_RVOL":rvol,
        })
    return pd.DataFrame(rows)


def option_skew_proxy(chain: pd.DataFrame,spot: float) -> tuple[float,pd.DataFrame]:
    """±4% OTM IV ratio proxy; Yahoo's public chain does not include option delta."""
    if chain.empty or not np.isfinite(spot): return np.nan,pd.DataFrame()
    nearest_expiry=sorted(chain["Expiration"].dropna().astype(str).unique())[0]
    subset=chain[chain["Expiration"].astype(str).eq(nearest_expiry)].copy()
    calls=subset[subset["Type"].eq("Call")].dropna(subset=["strike","impliedVolatility"])
    puts=subset[subset["Type"].eq("Put")].dropna(subset=["strike","impliedVolatility"])
    if calls.empty or puts.empty: return np.nan,pd.DataFrame()
    call_row=calls.iloc[(calls["strike"]-spot*1.04).abs().argsort()[:1]]
    put_row=puts.iloc[(puts["strike"]-spot*.96).abs().argsort()[:1]]
    put_iv=float(put_row["impliedVolatility"].iloc[0]); call_iv=float(call_row["impliedVolatility"].iloc[0])
    ratio=call_iv/put_iv if put_iv>0 else np.nan
    curve=subset[subset["strike"].between(spot*.75,spot*1.25)].copy()
    return ratio,curve


FLOW_RECORD_COLUMNS=["date","shares_outstanding","nav","source_url","source_note"]


def _normalise_flow_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a strict, duplicate-free issuer record table; malformed rows are ignored."""
    if frame is None or frame.empty: return pd.DataFrame(columns=FLOW_RECORD_COLUMNS)
    clean=frame.copy()
    for column in FLOW_RECORD_COLUMNS:
        if column not in clean: clean[column]=""
    clean["date"]=pd.to_datetime(clean["date"],errors="coerce").dt.tz_localize(None).dt.normalize()
    clean["shares_outstanding"]=pd.to_numeric(clean["shares_outstanding"],errors="coerce")
    clean["nav"]=pd.to_numeric(clean["nav"],errors="coerce")
    clean=clean.dropna(subset=["date","shares_outstanding","nav"])
    clean=clean[clean["shares_outstanding"].gt(0)&clean["nav"].gt(0)]
    return clean[FLOW_RECORD_COLUMNS].sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)


def _read_baseline_records(symbol: str) -> pd.DataFrame:
    data_dir=Path(__file__).resolve().parent/"data"; frames=[]
    for path in (data_dir/f"{symbol.lower()}_shares_baseline.csv",
                 data_dir/f"{symbol.lower()}_shares_history.csv",
                 data_dir/f"{symbol.lower()}_historical_shares.csv",
                 data_dir/f"{symbol.lower()}_flow_history.csv"):
        try:
            if path.exists() and path.stat().st_size:
                frames.append(pd.read_csv(path))
        except Exception:
            continue
    json_path=data_dir/f"{symbol.lower()}_shares_baseline.json"
    try:
        if json_path.exists():
            payload=json.loads(json_path.read_text(encoding="utf-8"))
            frames.append(pd.DataFrame(payload.get("records",payload) if isinstance(payload,dict) else payload))
    except Exception:
        pass
    return _normalise_flow_records(pd.concat(frames,ignore_index=True) if frames else pd.DataFrame())


def _xlsx_first_sheet(payload: bytes) -> list[list[object]]:
    """Small dependency-free XLSX reader used for State Street's public SPY workbook."""
    namespace="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        shared=[]
        if "xl/sharedStrings.xml" in archive.namelist():
            root=ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{namespace}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{namespace}t")))
        sheet_name=next((name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")),None)
        if not sheet_name: return []
        root=ET.fromstring(archive.read(sheet_name)); rows=[]
        for row in root.iter(f"{namespace}row"):
            values=[]
            for cell in row.findall(f"{namespace}c"):
                ref=cell.attrib.get("r","A1"); letters="".join(ch for ch in ref if ch.isalpha())
                column=0
                for letter in letters: column=column*26+ord(letter.upper())-64
                while len(values)<column: values.append(None)
                node=cell.find(f"{namespace}v"); value=node.text if node is not None else None
                if cell.attrib.get("t")=="s" and value is not None:
                    try: value=shared[int(value)]
                    except Exception: pass
                elif cell.attrib.get("t")=="inlineStr":
                    value="".join(item.text or "" for item in cell.iter(f"{namespace}t"))
                values[column-1]=value
            rows.append(values)
        return rows


def _state_street_share_history(symbol: str) -> pd.DataFrame:
    """Parse an issuer NAV-history workbook; reject missing share/date columns."""
    url=f"https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-{symbol.lower()}.xlsx"
    response=requests.get(url,headers={"User-Agent":"Mozilla/5.0","Accept":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*"},timeout=18)
    response.raise_for_status()
    rows=_xlsx_first_sheet(response.content)
    header_index=next((index for index,row in enumerate(rows)
        if any(str(item).strip().lower()=="date" for item in row if item is not None)),None)
    if header_index is None:
        return pd.DataFrame(columns=FLOW_RECORD_COLUMNS)
    headers=[str(value).strip() if value is not None else "" for value in rows[header_index]]
    frame=pd.DataFrame(rows[header_index+1:])
    frame=frame.iloc[:,:len(headers)]
    frame.columns=headers[:len(frame.columns)]
    lookup={re.sub(r"[^a-z]","",column.lower()):column for column in frame.columns}
    date_col=lookup.get("date")
    shares_col=next((value for key,value in lookup.items() if "sharesoutstanding" in key),None)
    nav_col=lookup.get("nav")
    if not date_col or not shares_col or not nav_col:
        return pd.DataFrame(columns=FLOW_RECORD_COLUMNS)
    date_values=frame[date_col]
    numeric=pd.to_numeric(date_values,errors="coerce")
    parsed=pd.to_datetime(date_values.where(numeric.isna()),errors="coerce")
    parsed=parsed.fillna(pd.to_datetime(numeric,unit="D",origin="1899-12-30",errors="coerce"))
    return _normalise_flow_records(pd.DataFrame({"date":parsed,
        "shares_outstanding":pd.to_numeric(frame[shares_col].astype(str).str.replace(",","",regex=False),errors="coerce"),
        "nav":pd.to_numeric(frame[nav_col].astype(str).str.replace(",","",regex=False),errors="coerce"),
        "source_url":url,"source_note":"State Street official daily NAV/share history"}))


@st.cache_data(ttl=900,show_spinner=False)
def load_dual_rrg_share_records() -> dict[str,pd.DataFrame]:
    """Issuer observations only; local snapshots survive issuer outages."""
    output={ticker:_read_baseline_records(ticker) for ticker in FLOW_TICKERS}
    def fetch(ticker: str) -> tuple[str,pd.DataFrame]:
        try:
            return ticker,_state_street_share_history(ticker)
        except Exception:
            return ticker,pd.DataFrame(columns=FLOW_RECORD_COLUMNS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ticker,incoming in pool.map(fetch,SECTORS):
            if not incoming.empty:
                output[ticker]=_normalise_flow_records(pd.concat([output[ticker],incoming],ignore_index=True))
    return output


def _deep_find_number(obj: object, names: set[str]) -> float:
    if isinstance(obj,dict):
        for key,value in obj.items():
            normal=re.sub(r"[^a-z]","",str(key).lower())
            if normal in names:
                try: return float(str(value).replace(",","").replace("$",""))
                except Exception: pass
        for value in obj.values():
            found=_deep_find_number(value,names)
            if np.isfinite(found): return found
    elif isinstance(obj,list):
        for value in obj:
            found=_deep_find_number(value,names)
            if np.isfinite(found): return found
    return np.nan


@st.cache_data(ttl=900,show_spinner=False)
def fetch_issuer_flow_records(symbol: str) -> pd.DataFrame:
    """Fetch only issuer-published NAV/share observations; no price-volume proxy."""
    symbol=symbol.upper().strip(); headers={"User-Agent":"Mozilla/5.0 (compatible; BuySideQuantPortal/1.0)"}
    try:
        if symbol=="SOXX":
            url="https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf"
            page=html.unescape(requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=15).text)
            shares_match=re.search(r'"sharesOutstanding":(\{[^{}]*\})',page)
            nav_match=re.search(r'"name":"NAV as of","value":"([\d,.]+)"[^{}]*"valueReference":\{[^{}]*"value":"([^"]+)"',page)
            if shares_match and nav_match:
                share_data=json.loads(shares_match.group(1))
                share_date=pd.to_datetime(share_data.get("formattedAsOfDate"),errors="coerce")
                nav_date=pd.to_datetime(nav_match.group(2),errors="coerce")
                if pd.notna(share_date) and pd.notna(nav_date) and share_date.normalize()==nav_date.normalize():
                    return _normalise_flow_records(pd.DataFrame([{
                        "date":share_date,"shares_outstanding":str(share_data.get("formattedValue","")).replace(",",""),
                        "nav":nav_match.group(1).replace(",",""),"source_url":url,
                        "source_note":"iShares official SOXX shares and NAV",
                    }]))
        if symbol=="TQQQ":
            url="https://accounts.profunds.com/etfdata/ByFund/TQQQ-historical_nav.csv"
            raw=pd.read_csv(StringIO(urlopen(Request(url,headers=headers),timeout=20).read().decode("utf-8-sig",errors="replace")))
            return _normalise_flow_records(pd.DataFrame({
                "date":raw.get("Date"),"shares_outstanding":pd.to_numeric(raw.get("Shares Outstanding (000)"),errors="coerce")*1000,
                "nav":raw.get("NAV"),"source_url":url,"source_note":"ProShares official daily NAV file",
            }))
        if symbol=="SPY":
            url="https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-spy.xlsx"
            rows=_xlsx_first_sheet(urlopen(Request(url,headers=headers),timeout=25).read())
            header_index=next((i for i,row in enumerate(rows) if row and str(row[0]).strip().lower()=="date"),None)
            if header_index is None: return pd.DataFrame(columns=FLOW_RECORD_COLUMNS)
            header=[str(item).strip() if item is not None else "" for item in rows[header_index]]
            raw=pd.DataFrame(rows[header_index+1:],columns=header)
            return _normalise_flow_records(pd.DataFrame({
                "date":raw.get("Date"),"shares_outstanding":raw.get("Shares Outstanding"),"nav":raw.get("NAV"),
                "source_url":url,"source_note":"State Street official SPY NAV history",
            }))
        if symbol=="SOXL":
            url="https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs"
            page=urlopen(Request(url,headers=headers),timeout=25).read().decode("utf-8",errors="replace")
            match=re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',page,re.S|re.I)
            payload=json.loads(html.unescape(match.group(1))) if match else {}
            candidates=[]
            def collect(item: object) -> None:
                if isinstance(item,dict):
                    if str(item.get("Ticker",item.get("ticker",""))).upper()=="SOXL": candidates.append(item)
                    for value in item.values(): collect(value)
                elif isinstance(item,list):
                    for value in item: collect(value)
            collect(payload)
            fund=candidates[0] if candidates else {}
            shares=_deep_find_number(fund,{"sharesoutstanding","outstandingshares"})
            nav=_deep_find_number(fund,{"nav","netassetvalue"})
            raw_date=str(fund.get("Pricing",{}).get("TradeDate",fund.get("TradeDate",""))) if isinstance(fund,dict) else ""
            observed=pd.to_datetime(raw_date,format="%m%d%Y",errors="coerce")
            if np.isfinite(shares) and np.isfinite(nav) and pd.notna(observed):
                return _normalise_flow_records(pd.DataFrame([{"date":observed,"shares_outstanding":shares,"nav":nav,
                    "source_url":url,"source_note":"Direxion official product page"}]))
        if symbol=="QQQ":
            url=("https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/46090E103/prices"
                 "?idType=cusip&productType=ETF&variationType=priceListing&productSubType=ETF")
            api_headers={**headers,"User-Agent":"BuySideQuantPortal/1.0"}
            payload=json.loads(urlopen(Request(url,headers=api_headers),timeout=25).read().decode("utf-8",errors="replace"))
            return _normalise_flow_records(pd.DataFrame([{
                "date":payload.get("effectiveDate"),"shares_outstanding":payload.get("sharesOutstanding"),
                "nav":payload.get("nav"),"source_url":url,"source_note":"Invesco official QQQ prices API",
            }]))
    except Exception:
        pass
    return pd.DataFrame(columns=FLOW_RECORD_COLUMNS)


def _persist_latest_flow_record(symbol: str, records: pd.DataFrame) -> str:
    """Atomically append a new issuer observation; cloud ephemeral files fail safely."""
    if records.empty: return "NO NEW ISSUER SHARE RECORD"
    path=Path(__file__).resolve().parent/"data"/f"{symbol.lower()}_flow_history.csv"
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        existing=pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(columns=FLOW_RECORD_COLUMNS)
        combined=_normalise_flow_records(pd.concat([existing,records.tail(1)],ignore_index=True))
        if not existing.empty and len(combined)==len(_normalise_flow_records(existing)):
            return "LOCAL LOGGER CURRENT"
        temporary=path.with_suffix(".csv.tmp")
        combined.to_csv(temporary,index=False,date_format="%Y-%m-%d")
        temporary.replace(path)
        return "LOCAL LOGGER UPDATED"
    except Exception as exc:
        return f"LOGGER READ-ONLY / {type(exc).__name__}"


@st.cache_data(ttl=900,show_spinner=False)
def load_primary_flow(symbol: str) -> pd.DataFrame:
    """Merge issuer observations and price; zero display bars never imply verified flows."""
    symbol=symbol.upper().strip(); end=date.today()+timedelta(days=1); start=end-timedelta(days=550)
    try:
        history=yf.Ticker(symbol).history(start=start,end=end,auto_adjust=True)[["Close"]].dropna(subset=["Close"])
    except Exception:
        history=pd.DataFrame()
    if history.empty: return pd.DataFrame()
    history.index=pd.to_datetime(history.index)
    if history.index.tz is not None: history.index=history.index.tz_convert(None)
    history=history.groupby(history.index.normalize()).last().sort_index()

    baseline=_read_baseline_records(symbol); live=fetch_issuer_flow_records(symbol)
    logger_status=_persist_latest_flow_record(symbol,live)
    logged=_read_baseline_records(symbol)
    records=_normalise_flow_records(pd.concat([baseline,logged,live],ignore_index=True))
    if not records.empty: records=records.set_index("date")

    data=history.copy(); data["HasOfficialRecord"]=data.index.isin(records.index) if not records.empty else False
    if records.empty:
        data["Shares"]=np.nan; data["NAV"]=np.nan; data["DailyFlow"]=0.0
        data["Source"]="NO VERIFIED ISSUER SHARES DATA"; data["OfficialAsOf"]=pd.NaT
        data["HasVerifiedShares"]=False; data["HasVerifiedFlow"]=False
        data["LoggerStatus"]=logger_status; data["RecordCount"]=0
        return data.tail(180)

    data["Shares"]=records["shares_outstanding"].reindex(data.index).ffill()
    data["NAV"]=records["nav"].reindex(data.index)
    prior_nav=records["nav"].shift(1)
    reported_delta=records["shares_outstanding"].diff()
    # A sparse baseline must never collapse weeks/months of creations into one fake daily spike.
    adjacent=records.index.to_series().diff().dt.days.le(5)
    record_flow=(reported_delta*prior_nav).where(adjacent)
    data["DailyFlow"]=record_flow.reindex(data.index).where(data["HasOfficialRecord"],0.0).fillna(0.0)
    data["HasVerifiedFlow"]=data.index.isin(record_flow.dropna().index)
    data["OfficialAsOf"]=records.index.max(); data["HasVerifiedShares"]=True
    source_notes=records["source_note"].dropna().astype(str); source_notes=source_notes[source_notes.str.len().gt(0)]
    data["Source"]=" + ".join(dict.fromkeys(source_notes.tail(4))) or "LOCAL VERIFIED BASELINE"
    data["LoggerStatus"]=logger_status; data["RecordCount"]=len(records)
    return data.dropna(subset=["Close"]).tail(180)


def sector_rs_scoreboard(price_frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    sector_specs={**SECTORS,"SMH":("Semiconductors","VanEck",0),"SOXX":("Semiconductors","iShares",0)}
    for ticker,(name,cn_name,_) in sector_specs.items():
        if ticker not in price_frame: continue
        pair=price_frame[[ticker,"SPY"]].dropna(); ratio=pair[ticker].div(pair["SPY"])
        if len(ratio)<200: continue
        latest=float(ratio.iloc[-1]); ma20=float(ratio.ewm(span=20,adjust=False).mean().iloc[-1])
        ma50=float(ratio.ewm(span=50,adjust=False).mean().iloc[-1]); ma200=float(ratio.rolling(200).mean().iloc[-1])
        above20,above50,above200=latest>ma20,latest>ma50,latest>ma200
        if above20 and above50 and above200: regime="🔥 Absolute Strength"
        elif above20 and above50 and not above200: regime="⚡ Counter-Trend Rebound"
        elif not above20 and not above50 and above200: regime="⚠️ Strong Pullback"
        elif not above20 and not above50 and not above200: regime="❄️ Weak Breakdown"
        else: regime="🔄 Transition"
        rows.append({"Sector":f"{name} · {cn_name}","ETF":ticker,"RS vs 20EMA":(latest/ma20-1)*100,
                     "RS vs 50EMA":(latest/ma50-1)*100,"RS vs 200SMA":(latest/ma200-1)*100,"Regime":regime})
    return pd.DataFrame(rows)


def rrg_improving_crossovers(price_frame: pd.DataFrame,tickers: list[str],benchmark: str="SPY") -> list[str]:
    transitions=[]
    if benchmark not in price_frame: return transitions
    for ticker in tickers:
        if ticker not in price_frame: continue
        frame=rrg_frame(price_frame[ticker],price_frame[benchmark]).tail(4)
        if len(frame)<4: continue
        current=frame.iloc[-1]; prior=frame.iloc[:-1]
        if current["rs_ratio"]<100 and current["rs_momentum"]>100 and ((prior["rs_ratio"]<100)&(prior["rs_momentum"]<100)).any():
            transitions.append(ticker)
    return transitions


def circuit_breaker_state(price_frame: pd.DataFrame) -> dict[str,object]:
    broken=[]
    for ticker in ("SPY","QQQ"):
        s=price_frame[ticker].dropna() if ticker in price_frame else pd.Series(dtype=float)
        sma=s.rolling(200,min_periods=200).mean()
        if not sma.dropna().empty and float(s.iloc[-1])<float(sma.iloc[-1]): broken.append(ticker)
    credit_warning=False; credit_change=np.nan
    if {"HYG","LQD"}.issubset(price_frame.columns):
        credit=price_frame[["HYG","LQD"]].dropna(); ratio=credit["HYG"].div(credit["LQD"])
        if len(ratio)>=50:
            credit_change=last_return(ratio,20); credit_warning=bool(ratio.iloc[-1]<ratio.rolling(50).mean().iloc[-1] and credit_change<0)
    vix_ratio=np.nan
    if {"^VIX","^VIX3M"}.issubset(price_frame.columns):
        vix_pair=price_frame[["^VIX","^VIX3M"]].dropna()
        if not vix_pair.empty and vix_pair.iloc[-1,1]>0: vix_ratio=float(vix_pair.iloc[-1,0]/vix_pair.iloc[-1,1])
    return {"broken":broken,"credit_warning":credit_warning,"credit_change":credit_change,
            "vix_ratio":vix_ratio,"vix_inverted":bool(np.isfinite(vix_ratio) and vix_ratio>=1)}


def last_return(series: pd.Series, periods: int) -> float:
    s = series.dropna()
    return np.nan if len(s) <= periods else float((s.iloc[-1] / s.iloc[-periods - 1] - 1) * 100)


def ytd_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    y = s[s.index.year == s.index[-1].year]
    return np.nan if y.empty or y.iloc[0] == 0 else float((y.iloc[-1] / y.iloc[0] - 1) * 100)


def drawdown_from_high(series: pd.Series) -> float:
    s = series.dropna()
    return np.nan if s.empty else float((s.iloc[-1] / s.max() - 1) * 100)


def normalized_slope(series: pd.Series) -> float:
    sma = series.dropna().rolling(200, min_periods=200).mean().dropna().tail(20)
    if len(sma) < 20 or sma.iloc[-1] == 0:
        return np.nan
    return float(np.polyfit(np.arange(20, dtype=float), sma.to_numpy(dtype=float), 1)[0] / sma.iloc[-1] * 100)


def trend_label(slope: float) -> str:
    if not np.isfinite(slope): return "Insufficient data"
    if slope > .035: return "Bullish acceleration"
    if slope < -.035: return "Bearish deterioration"
    return "Flat / range-bound"


def rrg_frame(target: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    pair = pd.concat([target, benchmark], axis=1).dropna()
    if len(pair) < 111:
        return pd.DataFrame(columns=["rs_ratio", "rs_momentum"])
    ratio = pair.iloc[:, 0].div(pair.iloc[:, 1]).replace([np.inf, -np.inf], np.nan)
    rs_ratio = ratio.div(ratio.rolling(100, min_periods=100).mean()).mul(100)
    momentum = rs_ratio.div(rs_ratio.shift(10)).mul(100)
    return pd.DataFrame({"rs_ratio": rs_ratio, "rs_momentum": momentum}).dropna()


def rolling_zscore(target: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    pair = pd.concat([target, benchmark], axis=1).dropna()
    ratio = pair.iloc[:, 0].div(pair.iloc[:, 1]).replace([np.inf, -np.inf], np.nan)
    mean, std = ratio.rolling(window, min_periods=window).mean(), ratio.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return ratio.sub(mean).div(std).replace([np.inf, -np.inf], np.nan).dropna()


def breadth_history(price_frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Daily advance/decline statistics for the available tracked constituents."""
    usable = [t for t in tickers if t in price_frame and price_frame[t].notna().any()]
    if not usable:
        return pd.DataFrame(columns=["adv", "dec", "unch", "net_pct", "ad_line"])
    changes = price_frame[usable].ffill(limit=3).pct_change(fill_method=None)
    adv, dec = (changes > 0).sum(axis=1), (changes < 0).sum(axis=1)
    unchanged = (changes == 0).sum(axis=1)
    active = (adv + dec).replace(0, np.nan)
    net_pct = (adv - dec).div(active).mul(100)
    return pd.DataFrame({"adv": adv, "dec": dec, "unch": unchanged,
                         "net_pct": net_pct, "ad_line": (adv - dec).cumsum()})


def percent_above_ma(price_frame: pd.DataFrame, tickers: list[str], window: int) -> float:
    usable = [t for t in tickers if t in price_frame and price_frame[t].dropna().shape[0] >= window]
    if not usable:
        return np.nan
    panel = price_frame[usable].ffill(limit=3)
    current = panel.iloc[-1]
    average = panel.rolling(window, min_periods=window).mean().iloc[-1]
    valid = current.notna() & average.notna()
    signal = current[valid].gt(average[valid])
    return float(signal.mean() * 100) if not signal.empty else np.nan


def value_change_bps(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    return np.nan if len(clean) <= periods else float((clean.iloc[-1] - clean.iloc[-periods - 1]) * 100)


def display_window(data: pd.Series | pd.DataFrame, label: str) -> pd.Series | pd.DataFrame:
    """Clip only for display; indicators should be calculated on pre-roll data first."""
    if data.empty:
        return data
    if label == "YTD":
        return data[data.index.year == data.index[-1].year]
    bars = PERIOD_BARS[label]
    return data.tail(bars) if bars else data


def display_with_preroll(data: pd.Series | pd.DataFrame, label: str, preroll: int = 252) -> pd.Series | pd.DataFrame:
    """Show one extra trading year before the selected range for visible MA context."""
    if data.empty:
        return data
    if label == "YTD":
        active = data[data.index.year == data.index[-1].year]
        start_pos = max(0,data.index.searchsorted(active.index[0])-preroll) if not active.empty else 0
        return data.iloc[start_pos:]
    bars = PERIOD_BARS[label] or len(data)
    return data.tail(min(len(data),bars+preroll))


def market_sentiment(price_frame: pd.DataFrame) -> tuple[str,str,int]:
    """Classify risk appetite from transparent cross-asset relative trends."""
    checks=[]
    descriptions=[]
    pairs=[("QQQ","SPY","Technology vs broad market"),("IWM","SPY","Small caps vs broad market"),("HYG","LQD","High yield vs investment grade")]
    for numerator,denominator,label in pairs:
        if numerator in price_frame and denominator in price_frame:
            move=last_return(price_frame[numerator].div(price_frame[denominator]),20)
            if np.isfinite(move):
                checks.append(1 if move>0 else -1); descriptions.append(f"{label}{move:+.1f}%")
    for numerator,label in [("GLD","Gold vs SPY"),("TLT","Long duration vs SPY")]:
        if numerator in price_frame and "SPY" in price_frame:
            move=last_return(price_frame[numerator].div(price_frame["SPY"]),20)
            if np.isfinite(move):
                checks.append(-1 if move>0 else 1); descriptions.append(f"{label}{move:+.1f}%")
    if "^VIX" in price_frame:
        move=last_return(price_frame["^VIX"],20)
        if np.isfinite(move):
            checks.append(-1 if move>0 else 1); descriptions.append(f"VIX 1M change {move:+.1f}%")
    score=sum(checks)
    if score>=3: state="Risk Appetite Returning / Risk-On"
    elif score<=-3: state="Defensive Demand Rising / Risk-Off"
    else: state="Mixed Market / Transition"
    return state,"；".join(descriptions),score


def signed_flow_score(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    pair = pd.concat([close.rename("close"),volume.rename("volume")],axis=1).dropna()
    if len(pair) < window + 1:
        return np.nan
    dollar_volume = pair["close"].mul(pair["volume"])
    signed = np.sign(pair["close"].pct_change(fill_method=None)).mul(dollar_volume)
    denominator = dollar_volume.tail(window).sum()
    return np.nan if denominator == 0 else float(signed.tail(window).sum() / denominator * 100)


def macd_series(series: pd.Series, fast: int = 8, slow: int = 17, signal: int = 9) -> pd.DataFrame:
    clean = series.dropna()
    macd = clean.ewm(span=fast, adjust=False).mean() - clean.ewm(span=slow, adjust=False).mean()
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": signal_line, "hist": macd - signal_line})


def relative_signal(target: pd.Series, benchmark: pd.Series) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Build ratio indicators and a descriptive, non-predictive regime label."""
    pair = pd.concat([target, benchmark], axis=1).dropna()
    if len(pair) < 200:
        return pd.DataFrame(), {"state": "Insufficient data"}
    ratio = pair.iloc[:, 0].div(pair.iloc[:, 1]).replace([np.inf, -np.inf], np.nan).dropna()
    ma60 = ratio.rolling(60, min_periods=60).mean()
    ma200 = ratio.rolling(200, min_periods=200).mean()
    mean120 = ratio.rolling(120, min_periods=120).mean()
    std120 = ratio.rolling(120, min_periods=120).std(ddof=0).replace(0, np.nan)
    zscore = ratio.sub(mean120).div(std120)
    macd = macd_series(ratio)
    frame = pd.concat([ratio.rename("ratio"), ma60.rename("ma60"), ma200.rename("ma200"),
                       zscore.rename("zscore"), macd], axis=1).dropna(subset=["ratio"])
    latest = frame.dropna().iloc[-1] if not frame.dropna().empty else None
    if latest is None:
        return frame, {"state": "Insufficient data"}
    below_both = latest["ratio"] < latest["ma60"] and latest["ratio"] < latest["ma200"]
    above_both = latest["ratio"] > latest["ma60"] and latest["ratio"] > latest["ma200"]
    hist_rising = len(frame.dropna()) > 1 and latest["hist"] > frame.dropna().iloc[-2]["hist"]
    if above_both and latest["zscore"] >= 2:
        state = "Overheated · Consider trimming"
    elif above_both:
        state = "Strong · Hold / monitor"
    elif below_both and latest["zscore"] <= -1.5 and hist_rising:
        state = "Early watch · Bottoming candidate"
    elif below_both:
        state = "Early watch · Await stabilization"
    else:
        state = "Neutral · MA transition"
    stats: dict[str, float | str] = {
        "state": state, "ratio": float(latest["ratio"]), "zscore": float(latest["zscore"]),
        "vs_ma60": float((latest["ratio"] / latest["ma60"] - 1) * 100),
        "vs_ma200": float((latest["ratio"] / latest["ma200"] - 1) * 100),
        "macd_hist": float(latest["hist"]),
    }
    return frame, stats


def regime_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Wilder ADX/+DI/-DI plus trend and volatility indicators."""
    frame=ohlc[["Open","High","Low","Close","Volume"]].copy().dropna(subset=["High","Low","Close"])
    high,low,close=frame["High"],frame["Low"],frame["Close"]
    up_move=high.diff(); down_move=-low.diff()
    plus_dm=pd.Series(np.where((up_move>down_move)&(up_move>0),up_move,0.0),index=frame.index)
    minus_dm=pd.Series(np.where((down_move>up_move)&(down_move>0),down_move,0.0),index=frame.index)
    true_range=pd.concat([high.sub(low),high.sub(close.shift()).abs(),low.sub(close.shift()).abs()],axis=1).max(axis=1)
    atr=true_range.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    plus_di=100*plus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean().div(atr.replace(0,np.nan))
    minus_di=100*minus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean().div(atr.replace(0,np.nan))
    dx=100*plus_di.sub(minus_di).abs().div(plus_di.add(minus_di).replace(0,np.nan))
    frame["ATR14"]=atr; frame["+DI"]=plus_di; frame["-DI"]=minus_di
    frame["ADX14"]=dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    frame["EMA20"]=close.ewm(span=20,adjust=False).mean(); frame["EMA50"]=close.ewm(span=50,adjust=False).mean()
    frame["SMA200"]=close.rolling(200,min_periods=200).mean()
    middle=close.rolling(20,min_periods=20).mean(); std=close.rolling(20,min_periods=20).std(ddof=0)
    frame["BB_MID"]=middle; frame["BB_UPPER"]=middle.add(2*std); frame["BB_LOWER"]=middle.sub(2*std)
    frame["BBW"]=frame["BB_UPPER"].sub(frame["BB_LOWER"]).div(middle.replace(0,np.nan))
    return frame


def classify_regime(frame: pd.DataFrame) -> tuple[str,dict[str,float]]:
    valid=frame.dropna(subset=["ADX14","EMA20","EMA50","SMA200","BBW"])
    if valid.empty:
        return "Insufficient data",{}
    last=valid.iloc[-1]; close=float(last["Close"]); adx=float(last["ADX14"])
    prior_adx=float(valid["ADX14"].iloc[-6]) if len(valid)>=6 else adx
    high20=float(valid["High"].tail(20).max()); box_high=float(valid["High"].tail(30).max()); box_low=float(valid["Low"].tail(30).min())
    zone_low=min(float(last["EMA20"]),float(last["EMA50"])); zone_high=max(float(last["EMA20"]),float(last["EMA50"]))
    pullback=(close>float(last["EMA50"]) and close<=zone_high*1.02 and close>=zone_low*.985 and high20/close-1>=.02)
    if adx<22: state="📦 Range-Bound"
    elif close<float(last["EMA50"]) and adx>prior_adx: state="⚠️ Downtrend Break"
    elif pullback: state="📈 Bullish Pullback"
    elif close>float(last["EMA50"]) and float(last["+DI"])>float(last["-DI"]): state="📈 Bull Trend"
    else: state="🔄 Transition"
    return state,{"close":close,"adx":adx,"plus_di":float(last["+DI"]),"minus_di":float(last["-DI"]),
                  "ema20":float(last["EMA20"]),"ema50":float(last["EMA50"]),"sma200":float(last["SMA200"]),
                  "bbw":float(last["BBW"]),"box_high":box_high,"box_low":box_low,"atr":float(last["ATR14"])}


def detect_geometric_patterns(frame: pd.DataFrame) -> dict[str,object]:
    """Detect trend breaks, double bottoms, descending channels and converging triangles."""
    recent=frame.dropna(subset=["High","Low","Close"]).tail(120).copy()
    result:dict[str,object]={"recent":recent,"trend":None,"trend_breakout":False,
                             "double_bottom":None,"channel":None,"triangle":None}
    if len(recent)<30:
        return result
    highs=recent["High"].to_numpy(dtype=float); lows=recent["Low"].to_numpy(dtype=float); closes=recent["Close"].to_numpy(dtype=float)
    maxima=argrelextrema(highs,np.greater_equal,order=4)[0]; minima=argrelextrema(lows,np.less_equal,order=4)[0]
    maxima=maxima[(maxima>0)&(maxima<len(recent)-1)]; minima=minima[(minima>0)&(minima<len(recent)-1)]
    high_points=None
    for width in (3,2):
        for start in range(len(maxima)-width,-1,-1):
            candidate=maxima[start:start+width]
            if len(candidate)==width and np.all(np.diff(highs[candidate])<0):
                high_points=candidate; break
        if high_points is not None: break
    if high_points is not None:
        slope,intercept=np.polyfit(high_points,highs[high_points],1)
        x_line=np.arange(int(high_points[0]),len(recent)); y_line=slope*x_line+intercept
        current_line=float(slope*(len(recent)-1)+intercept); previous_line=float(slope*(len(recent)-2)+intercept)
        breakout=bool(closes[-1]>current_line and closes[-2]<=previous_line)
        result["trend"]={"idx":high_points,"x":x_line,"y":y_line,"level":current_line,"slope":float(slope)}
        result["trend_breakout"]=breakout
        low_points=None
        for width in (3,2):
            for start in range(len(minima)-width,-1,-1):
                candidate=minima[start:start+width]
                if len(candidate)==width and np.all(np.diff(lows[candidate])<0):
                    low_points=candidate; break
            if low_points is not None: break
        if low_points is not None and slope<0:
            lower_intercept=float(np.mean(lows[low_points]-slope*low_points))
            x0=int(min(high_points[0],low_points[0])); channel_x=np.arange(x0,len(recent))
            result["channel"]={"x":channel_x,"upper":slope*channel_x+intercept,
                               "lower":slope*channel_x+lower_intercept,"height":float(intercept-lower_intercept)}
    # Converging triangle: fit the latest 2-4 swing highs and lows independently.
    # A valid structure must narrow toward a future apex and retain price inside both rails.
    if len(maxima)>=2 and len(minima)>=2:
        triangle_candidates=[]
        for high_count in (4,3,2):
            for low_count in (4,3,2):
                if len(maxima)<high_count or len(minima)<low_count: continue
                hi_idx=maxima[-high_count:]; lo_idx=minima[-low_count:]
                first=max(int(hi_idx[0]),int(lo_idx[0])); last=len(recent)-1
                hi_slope,hi_intercept=np.polyfit(hi_idx,highs[hi_idx],1)
                lo_slope,lo_intercept=np.polyfit(lo_idx,lows[lo_idx],1)
                width_now=(hi_slope*last+hi_intercept)-(lo_slope*last+lo_intercept)
                width_start=(hi_slope*first+hi_intercept)-(lo_slope*first+lo_intercept)
                converging=hi_slope<lo_slope and width_start>0 and width_now>0 and width_now<width_start*.82
                apex=(lo_intercept-hi_intercept)/(hi_slope-lo_slope) if abs(hi_slope-lo_slope)>1e-12 else np.inf
                apex_ok=last<apex<=last+max(90,2*(last-first))
                upper_now=hi_slope*last+hi_intercept; lower_now=lo_slope*last+lo_intercept
                inside=lower_now*.985<=closes[-1]<=upper_now*1.015
                high_fit=np.mean(np.abs(highs[hi_idx]-(hi_slope*hi_idx+hi_intercept)))
                low_fit=np.mean(np.abs(lows[lo_idx]-(lo_slope*lo_idx+lo_intercept)))
                fit_ok=(high_fit+low_fit)/max(np.mean(closes),1e-9)<.035
                if converging and apex_ok and inside and fit_ok:
                    score=(high_count+low_count)*10-(width_now/width_start)*5-(high_fit+low_fit)
                    triangle_candidates.append((score,hi_idx,lo_idx,hi_slope,hi_intercept,lo_slope,lo_intercept,apex))
        if triangle_candidates:
            _,hi_idx,lo_idx,hi_slope,hi_intercept,lo_slope,lo_intercept,apex=max(triangle_candidates,key=lambda x:x[0])
            x0=int(min(hi_idx[0],lo_idx[0])); x_line=np.arange(x0,len(recent))
            upper=hi_slope*x_line+hi_intercept; lower=lo_slope*x_line+lo_intercept
            prior_upper=hi_slope*(len(recent)-2)+hi_intercept; prior_lower=lo_slope*(len(recent)-2)+lo_intercept
            upper_now=float(upper[-1]); lower_now=float(lower[-1])
            breakout_up=bool(closes[-1]>upper_now and closes[-2]<=prior_upper)
            breakout_down=bool(closes[-1]<lower_now and closes[-2]>=prior_lower)
            if hi_slope<0 and lo_slope>0: triangle_type="Symmetrical Triangle"
            elif abs(hi_slope)<=abs(lo_slope)*.30: triangle_type="Ascending Triangle"
            elif abs(lo_slope)<=abs(hi_slope)*.30: triangle_type="Descending Triangle"
            else: triangle_type="Converging Wedge"
            result["triangle"]={"x":x_line,"upper":upper,"lower":lower,"high_idx":hi_idx,"low_idx":lo_idx,
                                "type":triangle_type,"apex":float(apex),"breakout_up":breakout_up,
                                "breakout_down":breakout_down,"height":float(max(upper[0]-lower[0],0))}

    candidates=[]
    for a in range(len(minima)):
        for b in range(a+1,len(minima)):
            i,j=int(minima[a]),int(minima[b])
            if j-i<10 or j-i>80 or j<len(recent)-40: continue
            if abs(lows[i]-lows[j])/max(lows[i],lows[j])<=.015:
                peak_rel=int(np.argmax(highs[i:j+1])); peak=i+peak_rel; neckline=float(highs[peak])
                trough=max(float(lows[i]),float(lows[j]))
                # Require a visible middle rebound and keep the neckline between the two lows.
                if i<peak<j and neckline/trough-1>=.025:
                    candidates.append((j,i,peak,neckline))
    if candidates:
        _,i,peak,neckline=max(candidates,key=lambda x:x[0]); j=max(candidates,key=lambda x:x[0])[0]
        result["double_bottom"]={"idx":[i,peak,j],"prices":[float(lows[i]),float(highs[peak]),float(lows[j])],
                                 "neckline":neckline,"breakout":bool(closes[-1]>neckline),"low2":float(lows[j])}
    return result


def chart_initial_range(index: pd.Index, label: str) -> list[pd.Timestamp] | None:
    """Return only an initial viewport; underlying traces retain their full history."""
    if len(index)==0 or label=="MAX":
        return None
    offsets={"6M":pd.DateOffset(months=6),"1Y":pd.DateOffset(years=1),"3Y":pd.DateOffset(years=3)}
    end=pd.Timestamp(index[-1]); start=max(pd.Timestamp(index[0]),end-offsets[label])
    return [start,end]


def mover_frame(price_frame: pd.DataFrame, universe: dict[str, tuple]) -> pd.DataFrame:
    rows = []
    for ticker, meta in universe.items():
        if ticker not in price_frame:
            continue
        clean = price_frame[ticker].dropna()
        move = last_return(price_frame[ticker], 1)
        if clean.empty or not np.isfinite(move):
            continue
        group_key = meta[1]
        group = SECTORS[group_key][1] if group_key in SECTORS else group_key
        rows.append({"Ticker": ticker, "Name": meta[0], "Group": group,
                     "Last": float(clean.iloc[-1]), "1D %": move})
    return pd.DataFrame(rows)


def _secret(name: str, fallback: str = "") -> str:
    """Read an optional Streamlit secret without requiring a local secrets.toml."""
    try:
        return str(st.secrets.get(name,fallback))
    except Exception:
        return fallback


def send_webhook_notification(channel: str, message: str, webhook_url: str = "",
                              bot_token: str = "", chat_id: str = "") -> tuple[bool,str]:
    """Send one short notification; credentials never enter cache or disk."""
    try:
        if channel == "Telegram Bot":
            if not bot_token.strip() or not chat_id.strip():
                return False,"Enter the Bot Token and Chat ID."
            endpoint=f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
            response=requests.post(endpoint,json={"chat_id":chat_id.strip(),"text":message,
                "parse_mode":"Markdown","disable_web_page_preview":True},timeout=3)
        elif channel == "Discord Webhook":
            if not webhook_url.strip(): return False,"Enter the Webhook URL."
            response=requests.post(webhook_url.strip(),json={"username":"Buy-Side Quant Portal",
                "embeds":[{"title":"◈ QUANT ALERT CENTER","description":message,"color":15158332,
                           "footer":{"text":"Automated research signal · Not investment advice"}}]},timeout=3)
        else:
            if not webhook_url.strip(): return False,"Enter the Webhook URL."
            url=webhook_url.strip()
            if "open.feishu.cn" in url or "larksuite.com" in url:
                payload={"msg_type":"text","content":{"text":message}}
            else:
                payload={"text":message,"content":message,"message":message}
            response=requests.post(url,json=payload,timeout=3)
        response.raise_for_status()
        return True,f"HTTP {response.status_code} · Notification sent"
    except requests.RequestException as exc:
        status=getattr(getattr(exc,"response",None),"status_code",None)
        return False,"Notification failed"+(f" (HTTP {status})" if status else " (network timeout or unreachable endpoint)")
    except Exception:
        return False,"Notification failed (invalid configuration)"


def evaluate_market_alerts(breaker_state: dict[str,object], flows: dict[str,pd.DataFrame],
                           skews: dict[str,float], option_data: dict[str,tuple[pd.DataFrame,float]],
                           improving_names: list[str], asof: object) -> list[dict[str,object]]:
    """Central, deterministic alert matrix used by the briefing and webhook dispatcher."""
    alerts=[]; asof_text=str(pd.Timestamp(asof).date()) if pd.notna(asof) else str(date.today())
    def add(level: int, category: str, message: str) -> None:
        identity=f"{asof_text}|{level}|{category}|{message}"
        alerts.append({"ID":hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18],
            "Level":level,"Severity":{1:"🚨 L1 · CIRCUIT BREAKER",2:"🔥 L2 · HIGH CONVICTION",3:"⚡ L3 · TACTICAL"}[level],
            "Category":category,"Message":message,"As Of":asof_text})
    broken=list(breaker_state.get("broken",[]) or [])
    if broken: add(1,"200 SMA TREND LOCK",f"{', '.join(broken)} fell below the 200-day SMA; leveraged-long allocation is locked.")
    if bool(breaker_state.get("vix_inverted",False)):
        ratio=float(breaker_state.get("vix_ratio",np.nan)); add(1,"VIX TERM INVERSION",f"VIX/VIX3M = {ratio:.3f}; the volatility term structure is inverted.")
    for symbol,frame in flows.items():
        if frame.empty or "DailyFlow" not in frame: continue
        last=frame.iloc[-1]; value=float(last.get("DailyFlow",np.nan)); verified=bool(last.get("HasVerifiedShares",False))
        official=bool(last.get("HasOfficialRecord",False))
        if verified and official and np.isfinite(value) and value>=500_000_000:
            add(2,"PRIMARY ETF FLOW",f"{symbol} verified issuer shares imply ${value/1e6:,.0f}M of one-day net creation, above $500M.")
    for symbol,value in skews.items():
        if np.isfinite(value) and value>=.95: add(2,"OPTION IV SKEW",f"{symbol} Call/Put IV Skew = {value:.2f}, reaching the potential gamma-squeeze threshold.")
    tactical=[]
    for symbol,(chain,spot) in option_data.items():
        if chain.empty or not np.isfinite(spot): continue
        frame=chain.copy()
        for field in ("volume","openInterest","lastPrice","bid","ask","strike"):
            frame[field]=pd.to_numeric(frame.get(field,np.nan),errors="coerce")
        frame["ratio"]=frame["volume"].div(frame["openInterest"].replace(0,np.nan))
        frame["premium"]=frame["volume"].mul(frame["lastPrice"].fillna((frame["bid"]+frame["ask"])/2)).mul(100)
        frame=frame[(frame["strike"].sub(spot).abs().div(spot)<=.20)&(frame["ratio"]>=2.5)&
                    (frame["volume"]>=1000)&(frame["premium"]>=5_000_000)]
        if not frame.empty:
            top=frame.sort_values("premium",ascending=False).iloc[0]
            tactical.append(f"{symbol} {top.get('Type','Option')} {top['strike']:g} · Vol/OI {top['ratio']:.1f} · ${top['premium']/1e6:.1f}M")
    if tactical: add(3,"OPTIONS UOA","; ".join(tactical[:5]))
    if improving_names: add(3,"RRG ROTATION",f"{', '.join(improving_names[:8])} moved from Lagging into Improving over the last three sessions.")
    return alerts


STATIC_XRAY_HOLDINGS={
    "QQQ":[("NVDA","NVIDIA",8.8),("AAPL","Apple",8.1),("MSFT","Microsoft",7.4),("AVGO","Broadcom",5.0),
           ("AMZN","Amazon",5.3),("META","Meta",4.0),("GOOGL","Alphabet A",3.0),("GOOG","Alphabet C",2.8),
           ("TSLA","Tesla",3.3),("COST","Costco",1.2)],
    "SOXX":[("NVDA","NVIDIA",8.5),("AVGO","Broadcom",8.2),("AMD","AMD",7.3),("MU","Micron",6.5),
            ("LRCX","Lam Research",6.0),("KLAC","KLA",5.5),("AMAT","Applied Materials",5.3),("TSM","TSMC",4.3),
            ("ASML","ASML",4.0),("TXN","Texas Instruments",4.0)],
}


def _holdings_from_records(records: object) -> pd.DataFrame:
    candidates=[]
    def walk(node: object) -> None:
        if isinstance(node,list):
            if node and all(isinstance(item,dict) for item in node): candidates.append(node)
            for item in node: walk(item)
        elif isinstance(node,dict):
            for value in node.values(): walk(value)
    walk(records)
    rows=[]
    for candidate in candidates:
        for item in candidate:
            normal={re.sub(r"[^a-z]","",str(k).lower()):v for k,v in item.items()}
            ticker=next((normal[k] for k in normal if k in {"ticker","symbol","holdingticker"}),None)
            weight=next((normal[k] for k in normal if k in {"weight","weighting","weightpercent","percentofnetassets","percenttna"}),None)
            name=next((normal[k] for k in normal if k in {"name","company","holdingname","securityname"}),ticker)
            try: value=float(str(weight).replace("%","").replace(",",""))
            except Exception: continue
            if ticker and np.isfinite(value) and value>0: rows.append((str(ticker).strip(),str(name or ticker).strip(),value))
        if len(rows)>=5: break
    return pd.DataFrame(rows,columns=["Ticker","Name","Weight"])


@st.cache_data(ttl=3600,show_spinner=False)
def load_etf_holdings(fund: str) -> tuple[pd.DataFrame,str]:
    """Issuer-first top holdings with an explicit dated visual baseline fallback."""
    try:
        if fund=="QQQ":
            url="https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/46090E103/holdings/fund?idType=cusip&productType=ETF"
            response=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=10); response.raise_for_status()
            frame=_holdings_from_records(response.json())
            source="Invesco official holdings API"
        else:
            url="https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf/1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund"
            response=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=10); response.raise_for_status()
            text=response.content.decode("utf-8-sig",errors="ignore"); lines=text.splitlines()
            start=next(i for i,line in enumerate(lines) if line.startswith("Ticker,"))
            raw=pd.read_csv(StringIO("\n".join(lines[start:])))
            weight_col=next(c for c in raw if "Weight" in str(c)); name_col=next(c for c in raw if str(c).strip()=="Name")
            frame=raw.rename(columns={"Ticker":"Ticker",name_col:"Name",weight_col:"Weight"})[["Ticker","Name","Weight"]]
            frame["Weight"]=pd.to_numeric(frame["Weight"],errors="coerce")
            frame=frame[frame["Ticker"].astype(str).str.fullmatch(r"[A-Z][A-Z0-9.\-]*",na=False)]
            source="iShares official SOXX holdings CSV"
        frame=frame.dropna(subset=["Ticker","Weight"]).sort_values("Weight",ascending=False).head(20)
        if len(frame)<5: raise ValueError("issuer payload did not contain holdings")
        return frame.reset_index(drop=True),source
    except Exception:
        fallback=pd.DataFrame(STATIC_XRAY_HOLDINGS[fund],columns=["Ticker","Name","Weight"])
        return fallback,f"Built-in reference baseline ({fund}; issuer endpoint unavailable)"


@st.cache_data(ttl=3600,show_spinner=False)
def load_fred_series(series_ids: tuple[str,...]) -> tuple[pd.DataFrame,dict[str,str]]:
    """Merge current official observations with a dated, bundled offline snapshot."""
    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,text/csv,*/*;q=0.8",
    }
    snapshot_path=Path(__file__).resolve().parent/"data"/"fred_offline_snapshot.csv"
    try:
        snapshot=pd.read_csv(snapshot_path,index_col="DATE",parse_dates=["DATE"])
        snapshot=snapshot.apply(pd.to_numeric,errors="coerce").sort_index()
        snapshot=snapshot[~snapshot.index.isna()]
    except (OSError,ValueError,pd.errors.ParserError):
        snapshot=pd.DataFrame()

    def fetch_one(series_id: str) -> pd.Series:
        # OFR FSI is published by the Office of Financial Research, not as FRED's OFRFSI ID.
        url=("https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
             if series_id=="OFRFSI" else
             f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
        response=requests.get(url,headers=headers,timeout=10)
        response.raise_for_status()
        raw=pd.read_csv(StringIO(response.text))
        date_col="Date" if series_id=="OFRFSI" else raw.columns[0]
        value_col="OFR FSI" if series_id=="OFRFSI" else series_id
        if date_col not in raw or value_col not in raw:
            raise ValueError(f"unexpected CSV columns for {series_id}")
        dates=pd.to_datetime(raw[date_col],errors="coerce")
        values=pd.to_numeric(raw[value_col],errors="coerce")
        item=pd.Series(values.to_numpy(),index=dates,name=series_id).dropna()
        item=item[~item.index.isna()]
        if item.empty:
            raise ValueError(f"no numeric observations for {series_id}")
        return item[~item.index.duplicated(keep="last")].sort_index()

    live={}
    # A failed FRED endpoint should cost at most one timeout window, not eight in series.
    with ThreadPoolExecutor(max_workers=min(6,len(series_ids) or 1)) as pool:
        futures={pool.submit(fetch_one,series_id):series_id for series_id in series_ids}
        for future in as_completed(futures):
            try:
                live[futures[future]]=future.result()
            except Exception:
                # One malformed or blocked source must never prevent the other series rendering.
                pass
    combined={}; status={}
    for series_id in series_ids:
        offline=(snapshot[series_id].dropna() if series_id in snapshot else pd.Series(dtype=float))
        if series_id in live:
            combined[series_id]=live[series_id].combine_first(offline)
            status[series_id]="live"
        elif not offline.empty:
            combined[series_id]=offline
            status[series_id]="offline"
        else:
            status[series_id]="unavailable"
    frame=pd.concat(combined,axis=1,sort=False).sort_index() if combined else pd.DataFrame()
    return frame,status


def inflation_release_table(fred: pd.DataFrame) -> pd.DataFrame:
    monthly=fred.resample("ME").last() if not fred.empty else pd.DataFrame()
    result=pd.DataFrame(index=monthly.index)
    if "PCEPI" in monthly: result["Headline PCE · YoY"]=monthly["PCEPI"].pct_change(12,fill_method=None)*100
    if "PCEPILFE" in monthly: result["Core PCE · YoY"]=monthly["PCEPILFE"].pct_change(12,fill_method=None)*100
    if "PCETRIM1M158SFRBDAL" in monthly:
        one=monthly["PCETRIM1M158SFRBDAL"]
        result["Trimmed Mean PCE · 1M ann."]=one
        result["Trimmed Mean PCE · 6M ann."]=(one.div(100).add(1).pow(1/12).rolling(6).apply(np.prod,raw=True).pow(2)-1)*100
    if "PCETRIM12M159SFRBDAL" in monthly: result["Trimmed Mean PCE · 12M"]=monthly["PCETRIM12M159SFRBDAL"]
    if "PI" in monthly: result["Personal Income · MoM"]=monthly["PI"].pct_change(fill_method=None)*100
    if "PCE" in monthly: result["Personal Spending · MoM"]=monthly["PCE"].pct_change(fill_method=None)*100
    result=result.dropna(how="all").tail(12).T
    result.columns=[stamp.strftime("%Y-%m") for stamp in result.columns]
    return result


def heat_style_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_colors(row: pd.Series) -> list[str]:
        values=pd.to_numeric(row,errors="coerce"); finite=values[np.isfinite(values)]
        if finite.empty: return ["color:#718198" for _ in row]
        low,high=float(finite.min()),float(finite.max()); span=max(high-low,1e-9)
        styles=[]
        for value in values:
            if not np.isfinite(value): styles.append("color:#718198;background:#0b121d"); continue
            position=(float(value)-low)/span
            if position>.66: styles.append("color:#ffd6dc;background:rgba(151,35,57,.55);font-weight:700")
            elif position<.34: styles.append("color:#c9ffe9;background:rgba(20,112,82,.48);font-weight:700")
            else: styles.append("color:#d7e0ec;background:#101823")
        return styles
    return frame.style.format("{:.2f}",na_rep="—").apply(row_colors,axis=1)


def base_layout(fig: go.Figure, height: int, hovermode: str = "closest") -> go.Figure:
    current_theme = st.session_state.get("portal_theme_choice", "Bloomberg Pro")
    return apply_quant_theme(fig, height, hovermode, theme_name=current_theme)


def section_header(kicker: str, title: str, note: str) -> None:
    st.markdown(f'<div class="module"><div class="kicker">{kicker}</div><div class="module-title">{title}</div><div class="module-note">{note}</div></div>', unsafe_allow_html=True)


def color_return(value: object) -> str:
    if not isinstance(value, (float, int, np.floating)) or not np.isfinite(value): return "color:#718198"
    return "color:#31d6a0;font-weight:600" if value >= 0 else "color:#ff5874;font-weight:600"


def format_num(value: float, suffix: str = "%") -> str:
    return "N/A" if not np.isfinite(value) else f"{value:+.2f}{suffix}"


live_pulse_auto_refresh=True
tail_length=5
rs_horizon="1Y"
z_window=120
alert_channel="Discord Webhook"
alert_webhook_url=""
alert_bot_token=""
alert_chat_id=""
alert_auto_send=False
with st.sidebar:
    st.markdown("### ◈ QUANT CONTROLS")
    st.caption("Research cache 15 minutes · live pulse cache 45 seconds")
    live_pulse_auto_refresh=st.toggle("Live pulse · 60s partial refresh",value=True,
        help="Refreshes only Global Market Pulse; heavy RRG and macro requests remain cached.")
    tail_length = st.slider("RRG tail · trading days", 5, 10, 5)
    rs_horizon = st.select_slider("RS horizon", options=list(PERIOD_BARS), value="1Y")
    z_window = st.slider("Z-score window", 60, 250, 120, 5)
    with st.expander("🔔 Alert push settings",expanded=False):
        alert_channel=st.selectbox("Webhook type",["Discord Webhook","Telegram Bot","Generic Webhook"],key="alert_channel")
        if alert_channel=="Telegram Bot":
            alert_bot_token=st.text_input("Bot Token",value=_secret("TELEGRAM_BOT_TOKEN"),type="password",key="alert_bot_token")
            alert_chat_id=st.text_input("Chat ID",value=_secret("TELEGRAM_CHAT_ID"),key="alert_chat_id")
        else:
            alert_webhook_url=st.text_input("Webhook URL",value=_secret("ALERT_WEBHOOK_URL"),type="password",key="alert_webhook_url")
        alert_auto_send=st.toggle("Automatically send L1 / L2",value=False,key="alert_auto_send",
            help="Sends only alerts not already delivered during this browser session.")
        if st.button("📨 Send test notification",width="stretch"):
            test_message="**TEST PING**\nMacro Quant Desk alert channel is connected.\n`Research only · Not investment advice`"
            ok,note=send_webhook_notification(alert_channel,test_message,alert_webhook_url,alert_bot_token,alert_chat_id)
            (st.success if ok else st.error)(note)
    st.divider()
    st.caption("English research terminal · full workflow")

core_tickers = ["SPY", "QQQ", "IWM", "^GSPC", "^NDX", "^VIX", "^VIX3M", "HYG", "LQD"]
theme_tickers=[ticker for members in AI_THEMES.values() for ticker in members]
all_tickers = tuple(dict.fromkeys(core_tickers + list(SECTORS) + list(RRG_EXTRA) + list(COMMODITIES)
                                  + list(HEAT_UNIVERSE) + list(NASDAQ_UNIVERSE) + list(YIELD_TICKERS.values())
                                  + theme_tickers + list(PRIMARY_FLOW_ETFS)))
end_date, start_date = date.today() + timedelta(days=1), date.today() - timedelta(days=2300)
data_error=None
try:
    with st.spinner("Synchronizing market data…"):
        prices = load_prices(all_tickers, start_date, end_date)
except Exception as exc:
    data_error=f"{type(exc).__name__}: {exc}"
    prices=pd.DataFrame(columns=all_tickers)
if prices.empty or prices.get("SPY", pd.Series(dtype=float)).dropna().empty:
    data_error=data_error or "SPY benchmark data is unavailable."
    st.error(f"Market data unavailable: {data_error}")
    st.info("Refresh after checking the network. Yahoo Finance may temporarily rate-limit requests.")
    st.stop()

valid_columns = [c for c in prices if prices[c].notna().any()]
missing_tickers = [c for c in all_tickers if c not in valid_columns]
latest_date = prices.dropna(how="all").index.max() if not prices.dropna(how="all").empty else pd.NaT

# Exact company-level S&P breadth: 500 companies, with unavailable quotes shown separately.
sp500_symbols: tuple[str, ...] = ()
sp500_breadth_prices = pd.DataFrame()
try:
    sp500_symbols = load_sp500_companies()
    sp500_breadth_prices = load_sp500_breadth_prices(sp500_symbols)
except Exception:
    pass
if len(sp500_symbols) == 500 and not sp500_breadth_prices.empty:
    sp_changes = sp500_breadth_prices.ffill(limit=2).pct_change(fill_method=None).iloc[-1]
    sp_adv = int((sp_changes > 0).sum()); sp_dec = int((sp_changes < 0).sum())
    sp_unch = int((sp_changes == 0).sum()); sp_missing = 500 - sp_adv - sp_dec - sp_unch
else:
    sp_adv = sp_dec = sp_unch = 0; sp_missing = 500

sector_flow_close,sector_flow_volume=pd.DataFrame(),pd.DataFrame()
try:
    sector_flow_close,sector_flow_volume=load_flow_data(tuple(SECTORS),"3mo")
except Exception:
    pass

breakers=circuit_breaker_state(prices)


@st.fragment(run_every="60s" if live_pulse_auto_refresh else None)
def render_live_market_pulse() -> None:
    """Refresh only the header tape; heavy research modules stay outside this fragment."""
    session=us_market_status(); eastern=pytz.timezone("US/Eastern")
    refreshed_at=datetime.now(eastern)
    try:
        pulse=fetch_live_pulse()
    except Exception:
        pulse=pd.DataFrame()
    quote_stamps=[]
    if not pulse.empty and "QuoteTimeET" in pulse:
        quote_stamps=[stamp for stamp in pulse["QuoteTimeET"].tolist() if pd.notna(stamp)]
    quote_asof=max(quote_stamps).strftime("%H:%M:%S ET") if quote_stamps else "DAILY FALLBACK"
    st.markdown(f"""<div class="portal-head"><div><div class="portal-title">BUY-SIDE QUANT PORTAL</div>
    <div class="portal-sub">CROSS-ASSET INTELLIGENCE · ROTATION · PRIMARY FLOW · SYSTEM RISK</div></div>
    <div class="market-clock"><span style="color:{session['color']}">{session['label']}</span><small>{session['time']} · {session['countdown']}</small></div>
    <div class="live">LAST UPDATED {refreshed_at:%H:%M:%S} ET<br><span style="color:#71849a">QUOTE {quote_asof}</span></div></div>
    <div class="deck-label">GLOBAL MARKET PULSE · 60s FRAGMENT · 45s QUOTE CACHE</div>""",unsafe_allow_html=True)

    deck=st.columns(8)
    for idx,(display_ticker,source_ticker) in enumerate((("SPY","SPY"),("QQQ","QQQ"),("IWM","IWM"),("VIX","^VIX"))):
        live_price=live_change=np.nan
        if not pulse.empty and display_ticker in pulse.index:
            live_price=float(pulse.loc[display_ticker,"Price"]); live_change=float(pulse.loc[display_ticker,"ChangePct"])
        if not np.isfinite(live_price):
            fallback=prices[source_ticker].dropna() if source_ticker in prices else pd.Series(dtype=float)
            live_price=float(fallback.iloc[-1]) if not fallback.empty else np.nan; live_change=last_return(fallback,1)
        value="N/A" if not np.isfinite(live_price) else (f"{live_price:,.2f}" if display_ticker=="VIX" else f"${live_price:,.2f}")
        deck[idx].metric(display_ticker,value,format_num(live_change),delta_color="inverse" if display_ticker=="VIX" else "normal",
            help="Uses the latest intraday quote when available, then falls back to the most recent daily close. VIX uses inverse risk coloring.")

    spx_dd=drawdown_from_high(prices["^GSPC"].dropna().tail(504)) if "^GSPC" in prices else np.nan
    ndx_dd=drawdown_from_high(prices["^NDX"].dropna().tail(504)) if "^NDX" in prices else np.nan
    deck[4].metric("SPX DRAWDOWN",format_num(spx_dd),"FROM 2Y HIGH",delta_color="off",help="Drawdown from the highest daily close in the trailing two years.")
    deck[5].metric("NDX DRAWDOWN",format_num(ndx_dd),"FROM 2Y HIGH",delta_color="off",help="Drawdown from the highest daily close in the trailing two years.")
    sp_breadth_text="N/A" if sp_missing==500 else f"{sp_adv}/{sp_dec}/{sp_unch}/{sp_missing}"
    deck[6].metric("SPY 500 · A/D/U/M",sp_breadth_text,"SUM = 500",delta_color="off",
        help="Daily breadth across 500 constituents: advancing / declining / unchanged / missing. The four values always sum to 500.")
    ndx_returns=pd.Series({ticker:last_return(prices[ticker],1) for ticker in NASDAQ_UNIVERSE if ticker in prices}).dropna()
    ndx_adv,ndx_dec,ndx_unch=int((ndx_returns>0).sum()),int((ndx_returns<0).sum()),int((ndx_returns==0).sum())
    deck[7].metric("NDX SAMPLE · A / D / U",f"{ndx_adv} / {ndx_dec} / {ndx_unch}",
        f"{len(NASDAQ_UNIVERSE)-len(ndx_returns)} MISSING",delta_color="off",help="Breadth across the Nasdaq coverage universe used by this dashboard.")


render_live_market_pulse()

if breakers["broken"]:
    st.markdown(f'<div class="risk-lock">🚨 STRATEGIC RISK LOCK: {", ".join(breakers["broken"])} below the 200-day SMA. Leveraged-long allocation is disabled.</div>',unsafe_allow_html=True)

if missing_tickers:
    st.caption(f"⚠ Symbols skipped because no valid market data was returned: {', '.join(missing_tickers)}")

tab_briefing,tab_rotation,tab_market,tab_heat,tab_macro,tab_technical,tab_pattern,tab_flow,tab_circuit,tab_fixed,tab_xray,tab_deep,tab_pulse = st.tabs([
    "00  BRIEFING", "01  SECTOR & THEMES RRG",
    "02  MARKETS & BREADTH",
    "03  HEAT MAPS", "04  MACRO & COMMODITIES",
    "05  TECHNICALS", "06  REGIME & PATTERN",
    "07  PRIMARY ETF FLOW", "08  CIRCUIT BREAKERS", "09  FIXED INCOME",
    "10  ETF X-RAY", "11  MACRO DEEP DIVE", "12  EVENT PULSE",
])

with tab_briefing:
    section_header("DAILY COMMAND CENTER","00 Briefing","Cross-asset signal summary. Missing upstream data degrades only the affected signal and never blocks the rest of the terminal.")
    flow_snapshots={}
    # Briefing needs only the two leveraged products; QQQ/SPY are fetched lazily
    # when selected in Tab 08 so their issuer workbooks do not slow every rerun.
    for symbol in LEVERAGED_ETFS:
        try: flow_snapshots[symbol]=load_primary_flow(symbol)
        except Exception: flow_snapshots[symbol]=pd.DataFrame()
    # Option-chain features are intentionally excluded from the public English edition.
    option_snapshots={}
    latest_flows={symbol:(float(frame["DailyFlow"].dropna().iloc[-1]) if not frame.empty and not frame["DailyFlow"].dropna().empty else np.nan)
                  for symbol,frame in flow_snapshots.items()}
    flow_verified={symbol:(bool(frame["HasVerifiedShares"].iloc[-1]) if not frame.empty and "HasVerifiedShares" in frame else False)
                   for symbol,frame in flow_snapshots.items()}
    extreme_flow=[s for s in LEVERAGED_ETFS if np.isfinite(latest_flows.get(s,np.nan))
                  and latest_flows[s]>=500_000_000 and flow_verified.get(s,False)]
    skew_values={}
    rotation_universe=list(dict.fromkeys(list(SECTORS)+theme_tickers))
    improving=rrg_improving_crossovers(prices,rotation_universe)
    firewall_safe=not breakers["broken"] and not breakers["credit_warning"] and not breakers["vix_inverted"]
    alert_cards=st.columns(4)
    unavailable_flows=[symbol for symbol in LEVERAGED_ETFS if not flow_verified.get(symbol,False)]
    alert_cards[0].metric("🚨 PRIMARY-MARKET FLOW", "🔥 INSTITUTIONAL CREATION" if extreme_flow else "OFFICIAL FLOW WATCH",
        ", ".join(f"{s} ${latest_flows[s]/1e6:,.0f}M" for s in extreme_flow) if extreme_flow else
        (f"Awaiting issuer shares: {', '.join(unavailable_flows)}" if unavailable_flows else "TQQQ / SOXL · $500M threshold"),delta_color="off")
    alert_cards[1].metric("⚡ MARKET INTERNALS", "BREADTH CONFIRMED" if improving and firewall_safe else "MIXED / MONITOR",
        f"{len(improving)} recent RRG upgrades · {'Risk locks clear' if firewall_safe else 'Risk lock active'}",delta_color="off")
    alert_cards[2].metric("🎯 RRG IMPROVING", ", ".join(improving[:5]) if improving else "NO CONFIRMED CROSSOVER",
        "Lagging → Improving over the last 3 sessions",delta_color="off")
    alert_cards[3].metric("🛡️ MACRO FIREWALL", "🟢 STRATEGIC ENVIRONMENT CLEAR" if firewall_safe else "🚨 RISK CIRCUIT BREAKER",
        f"200SMA {len(breakers['broken'])} · Credit {'RISK' if breakers['credit_warning'] else 'OK'} · VIX {'INV' if breakers['vix_inverted'] else 'OK'}",delta_color="off")
    ai_momentum=[]
    for theme,members in AI_THEMES.items():
        if theme.startswith("Macro"): continue
        returns={ticker:last_return(prices[ticker],5) for ticker in members if ticker in prices}
        valid={ticker:value for ticker,value in returns.items() if np.isfinite(value)}
        if valid:
            leader=max(valid,key=valid.get); ai_momentum.append((theme,float(np.mean(list(valid.values()))),leader,valid[leader]))
    ai_momentum.sort(key=lambda item:item[1],reverse=True)
    section_header("QUANT DIAGNOSTIC","Daily Quantitative Diagnostic","A compact synthesis of leadership, systemic risk and verified primary-market activity.")
    ai_text="; ".join(f"{theme}: {leader} {leader_ret:+.1f}% (theme 5D mean {avg:+.1f}%)" for theme,avg,leader,leader_ret in ai_momentum[:3]) or "Insufficient AI-theme data"
    st.markdown(f"""
- **AI momentum:** {ai_text}
- **System risk:** index breakdowns `{', '.join(breakers['broken']) or 'none'}`; credit liquidity `{'tightening' if breakers['credit_warning'] else 'normal'}`; VIX term structure `{'inverted' if breakers['vix_inverted'] else 'contango / normal'}`.
- **Execution discipline:** signal conviction rises only when trend, breadth and verified primary-market creation agree. No single proxy is a trade instruction.
""")

    active_alerts=evaluate_market_alerts(breakers,flow_snapshots,skew_values,option_snapshots,improving,latest_date)
    existing_log=st.session_state.setdefault("_active_alert_log",[])
    known_ids={item.get("ID") for item in existing_log}
    for item in active_alerts:
        if item["ID"] not in known_ids:
            existing_log.insert(0,{**item,"Unread":True})
    st.session_state["_active_alert_log"]=existing_log[:100]

    if alert_auto_send:
        priority=[item for item in active_alerts if int(item["Level"])<=2]
        sent_ids=set(st.session_state.get("_sent_alert_ids",[])); pending=[item for item in priority if item["ID"] not in sent_ids]
        if pending:
            body="**BUY-SIDE QUANT PORTAL · ACTIVE ALERTS**\n"+"\n".join(
                f"{item['Severity']} | **{item['Category']}**\n{item['Message']}" for item in pending)
            ok,note=send_webhook_notification(alert_channel,body,alert_webhook_url,alert_bot_token,alert_chat_id)
            if ok:
                sent_ids.update(item["ID"] for item in pending); st.session_state["_sent_alert_ids"]=list(sent_ids)[-200:]
                st.toast(f"Sent {len(pending)} alert(s)",icon="🔔")
            elif alert_webhook_url or (alert_bot_token and alert_chat_id):
                st.warning(f"Automatic alert delivery failed: {note}")

    section_header("ALERT CENTER","Active Alerts Log","L1/L2 alerts may be sent by webhook. Duplicate rules for the same observation date are suppressed within the session.")
    log_frame=pd.DataFrame(st.session_state.get("_active_alert_log",[]))
    unread_count=int(log_frame["Unread"].sum()) if not log_frame.empty and "Unread" in log_frame else 0
    log_left,log_right=st.columns([5,1],vertical_alignment="bottom")
    log_left.caption(f"ACTIVE {len(active_alerts)} · UNREAD {unread_count} · HISTORY {len(log_frame)}")
    if log_right.button("Mark all as read",width="stretch",disabled=unread_count==0):
        for item in st.session_state.get("_active_alert_log",[]): item["Unread"]=False
        st.rerun()
    if log_frame.empty:
        st.success("🟢 No active alerts in the current rule matrix.")
    else:
        display_log=log_frame[["Unread","Severity","Category","Message","As Of"]].copy()
        display_log["Unread"]=display_log["Unread"].map({True:"● NEW",False:"READ"})
        st.dataframe(display_log,width="stretch",hide_index=True,height=min(330,72+35*len(display_log)))

with tab_rotation:
    rotation_head,rotation_control,benchmark_control=st.columns([4.6,1.75,1.0],vertical_alignment="bottom")
    with rotation_head:
        section_header("DUAL-RRG CONVERGENCE","01 Sector & Themes","Price rotation on the left; verified issuer-share creation and redemption on the right.")
    with rotation_control:
        selected_theme=st.selectbox("Theme universe",list(AI_THEMES),key="rrg_theme_selector")
    with benchmark_control:
        rrg_benchmark=st.selectbox("RRG Benchmark",["QQQ","SPY"],key="rrg_benchmark")
    selected_rrg_tickers=AI_THEMES[selected_theme]
    verified_shares=load_dual_rrg_share_records()
    flow_trails,flow_five_day=build_flow_rrg(prices,verified_shares)
    price_quadrants={}
    for ticker in FLOW_TICKERS:
        if ticker in prices and rrg_benchmark in prices:
            frame=rrg_frame(prices[ticker],prices[rrg_benchmark])
            if not frame.empty:
                last=frame.iloc[-1]
                price_quadrants[ticker]=flow_quadrant(float(last["rs_ratio"]),float(last["rs_momentum"]))
    convergence_rows=[]
    for ticker in FLOW_TICKERS:
        tail=flow_trails.get(ticker)
        flow_zone=flow_quadrant(float(tail["flow_ratio"].iloc[-1]),float(tail["flow_momentum"].iloc[-1])) if tail is not None else "N/A"
        price_zone=price_quadrants.get(ticker,"N/A")
        five_day=flow_five_day.get(ticker,np.nan)
        close=prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
        convergence_rows.append({"Sector":ticker,"Price":float(close.iloc[-1]) if not close.empty else np.nan,
            "Price Quadrant":price_zone,"Flow Quadrant":flow_zone,"5D Net Creation ($M)":five_day/1e6 if np.isfinite(five_day) else np.nan,
            "System Signal":convergence_label(price_zone,flow_zone,five_day)})
    convergence=pd.DataFrame(convergence_rows)
    early=convergence.loc[convergence["System Signal"].eq("🔥 机构抢跑起爆"),"Sector"].tolist()
    traps=convergence.loc[convergence["System Signal"].eq("⚠️ 缩量虚火诱多"),"Sector"].tolist()
    st.info(f"🔥 Institutional accumulation: {', '.join(early) or 'no verified signal'}  |  ⚠️ Price/flow divergence: {', '.join(traps) or 'no verified signal'}  |  Flow coverage: {len(flow_trails)}/{len(FLOW_TICKERS)} ETFs")
    theme_crossovers=rrg_improving_crossovers(prices,selected_rrg_tickers,rrg_benchmark)
    if theme_crossovers:
        st.success(f"🎯 Improving crossover: {', '.join(theme_crossovers)} moved from Lagging to Improving over the last three sessions.")
    else:
        st.info("No three-session Lagging → Improving crossover in the selected theme.")
    try:
        theme_ohlcv=load_ohlcv(tuple(selected_rrg_tickers),"1y")
    except Exception:
        theme_ohlcv=pd.DataFrame()
    ambush_candidates=[]; climax_candidates=[]
    for ticker in selected_rrg_tickers:
        if ticker not in prices or rrg_benchmark not in prices:
            continue
        try:
            tail=rrg_frame(prices[ticker],prices[rrg_benchmark])
            ticker_ohlcv=(theme_ohlcv[ticker] if isinstance(theme_ohlcv.columns,pd.MultiIndex)
                          and ticker in theme_ohlcv.columns.get_level_values(0) else pd.DataFrame())
            volume_trend=participation_metrics(ticker_ohlcv)["volume_trend_pct"]
            ambush,climax=early_rrg_signals(tail,prices[ticker],prices[rrg_benchmark],
                volume_trend_pct=volume_trend,five_day_official_flow=flow_five_day.get(ticker,np.nan))
            if ambush: ambush_candidates.append(ticker)
            if climax: climax_candidates.append(ticker)
        except (KeyError,ValueError,TypeError,IndexError):
            continue
    section_header("EARLY AMBUSH RADAR","Early Entry & Anti-Chase Radar",
        "RRG momentum + EMA20 + 5D/20D secondary-market participation. Verified 5D issuer flow is used when available; volume is never treated as fund flow.")
    if ambush_candidates:
        st.success(f"🎯 Early-entry candidates: {', '.join(ambush_candidates)} · momentum reversal plus participation / verified flow confirmation. Watchlist only.")
    else:
        st.info("No symbol currently meets the combined momentum reversal, EMA recovery and participation criteria.")
    if climax_candidates:
        st.warning(f"⚠️ Momentum exhaustion: {', '.join(climax_candidates)} · Leading-quadrant momentum is rolling over or RS-Ratio exceeds 105.")
    else:
        st.caption("No Leading-quadrant momentum-exhaustion signal in the selected theme.")
    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        section_header("PRICE RRG", f"Price Momentum Rotation · {selected_theme}", f"Relative to {rrg_benchmark}; the terminal point is the latest session.")
        fig, tails = go.Figure(), []
        rrg_names = {ticker:(SECTORS[ticker][1] if ticker in SECTORS else TACTICAL_NAMES.get(ticker,RRG_EXTRA.get(ticker,ticker))) for ticker in selected_rrg_tickers}
        palette=px.colors.qualitative.Bold+px.colors.qualitative.Safe
        for ticker, label in rrg_names.items():
            if ticker not in prices: continue
            tail = rrg_frame(prices[ticker], prices[rrg_benchmark]).tail(tail_length)
            if tail.empty: continue
            tails.append(tail)
            color = COLORS.get(ticker,palette[list(rrg_names).index(ticker)%len(palette)])
            fig.add_trace(go.Scatter(x=tail["rs_ratio"], y=tail["rs_momentum"], mode="lines+markers", name=ticker,
                showlegend=False, line=dict(color=color, width=1.8),
                marker=dict(color=color, size=np.linspace(3.5, 8.5, len(tail)), opacity=np.linspace(.3, 1, len(tail)), line=dict(color="#e9f1fb", width=.35)),
                customdata=tail.index.strftime("%Y-%m-%d"),
                hovertemplate=f"<b>{ticker} · {label}</b><br>%{{customdata}}<br>RS-Ratio %{{x:.2f}}<br>RS-Momentum %{{y:.2f}}<extra></extra>"))
            last = tail.iloc[-1]
            fig.add_annotation(x=float(last["rs_ratio"]), y=float(last["rs_momentum"]), text=f"<b>{ticker}</b>", showarrow=True,
                arrowhead=2, arrowsize=.9, arrowwidth=1.2, arrowcolor=color, ax=13, ay=-13, font=dict(color=color, size=9),
                bgcolor="rgba(5,9,15,.82)", bordercolor=color, borderwidth=.8, borderpad=2)
        if tails:
            mx, my = pd.concat([x["rs_ratio"] for x in tails]), pd.concat([x["rs_momentum"] for x in tails])
            xp, yp = max(.8, float(mx.max()-mx.min())*.15), max(.8, float(my.max()-my.min())*.15)
            xr, yr = [min(99, float(mx.min())-xp), max(101, float(mx.max())+xp)], [min(99, float(my.min())-yp), max(101, float(my.max())+yp)]
        else: xr, yr = [98, 102], [98, 102]
        base_layout(fig, 570)
        fig.add_vline(x=100, line_dash="dash", line_color="#71829a", line_width=1); fig.add_hline(y=100, line_dash="dash", line_color="#71829a", line_width=1)
        for x, y, text, color, anchor in [(.02,.97,"IMPROVING","#21d9a3","left"),(.98,.97,"LEADING","#ff5874","right"),(.02,.03,"LAGGING","#718cff","left"),(.98,.03,"WEAKENING","#e5c34d","right")]:
            fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=f"<b>{text}</b>", showarrow=False, xanchor=anchor, font=dict(color=color, size=9))
        fig.update_xaxes(title="RS-Ratio", range=xr); fig.update_yaxes(title="RS-Momentum", range=yr)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    with col_right:
        section_header("FLOW RRG", "Primary-Market Flow Rotation · Sectors + SMH/SOXX", "Cross-sectional 20D net creation / AUM with 5D velocity. Only consecutively disclosed issuer shares are used.")
        flow_fig=go.Figure()
        flow_tails=[]
        for index,ticker in enumerate(FLOW_TICKERS):
            tail=flow_trails.get(ticker)
            if tail is None: continue
            flow_tails.append(tail)
            color=COLORS.get(ticker,palette[index%len(palette)])
            flow_fig.add_trace(go.Scatter(x=tail["flow_ratio"],y=tail["flow_momentum"],mode="lines+markers",name=ticker,
                showlegend=False,line=dict(color=color,width=1.8),
                marker=dict(color=color,size=np.linspace(3.5,8.5,len(tail)),opacity=np.linspace(.3,1,len(tail)),line=dict(color="#e9f1fb",width=.35)),
                customdata=tail.index.strftime("%Y-%m-%d"),
                hovertemplate=f"<b>{ticker}</b><br>%{{customdata}}<br>Flow-Ratio %{{x:.2f}}<br>Flow-Momentum %{{y:.2f}}<extra></extra>"))
            last=tail.iloc[-1]
            flow_fig.add_annotation(x=float(last["flow_ratio"]),y=float(last["flow_momentum"]),text=f"<b>{ticker}</b>",showarrow=True,
                arrowhead=2,arrowsize=.9,arrowwidth=1.2,arrowcolor=color,ax=13,ay=-13,font=dict(color=color,size=9),
                bgcolor="rgba(5,9,15,.82)",bordercolor=color,borderwidth=.8,borderpad=2)
        if flow_tails:
            all_x=pd.concat([tail["flow_ratio"] for tail in flow_tails]); all_y=pd.concat([tail["flow_momentum"] for tail in flow_tails])
            x_pad=max(.8,float(all_x.max()-all_x.min())*.15); y_pad=max(.8,float(all_y.max()-all_y.min())*.15)
            flow_x=[min(99,float(all_x.min())-x_pad),max(101,float(all_x.max())+x_pad)]
            flow_y=[min(99,float(all_y.min())-y_pad),max(101,float(all_y.max())+y_pad)]
        else:
            flow_x,flow_y=[98,102],[98,102]
        base_layout(flow_fig,570)
        flow_fig.add_vline(x=100,line_dash="dash",line_color="#71829a",line_width=1)
        flow_fig.add_hline(y=100,line_dash="dash",line_color="#71829a",line_width=1)
        for x,y,label,color,anchor in [(.02,.97,"IMPROVING · Accumulation","#21d9a3","left"),(.98,.97,"LEADING · Institutional Demand","#ff5874","right"),
                                        (.02,.03,"LAGGING · Outflows","#718cff","left"),(.98,.03,"WEAKENING · Decelerating","#e5c34d","right")]:
            flow_fig.add_annotation(xref="paper",yref="paper",x=x,y=y,text=f"<b>{label}</b>",showarrow=False,xanchor=anchor,font=dict(color=color,size=9))
        flow_fig.update_xaxes(title="Flow-Intensity Ratio",range=flow_x)
        flow_fig.update_yaxes(title="Flow-Velocity Momentum",range=flow_y)
        st.plotly_chart(flow_fig,width="stretch",config={"displaylogo":False})
        if not flow_tails:
            st.caption("Insufficient consecutive issuer-share observations for a 20D intensity and 5D trail. Missing values are never backfilled with price or volume proxies.")

    signal_en={"🔥 机构抢跑起爆":"🔥 Institutional Accumulation","🚀 真金主升浪":"🚀 Confirmed Leadership","⚠️ 缩量虚火诱多":"⚠️ Price / Flow Divergence","❄️ 双弱阴跌深渊":"❄️ Dual Weakness","待核实·官方份额不足":"Pending · Insufficient Issuer Data"}
    convergence["System Signal"]=convergence["System Signal"].map(signal_en).fillna(convergence["System Signal"])
    section_header("CONVERGENCE MATRIX","Dual-Track Price / Flow Matrix","Dollar values in USD millions. N/A means issuer-share continuity is insufficient to infer primary-market creation or redemption.")
    st.dataframe(convergence.style.format({"Price":"${:,.2f}","5D Net Creation ($M)":"{:+,.1f}"},na_rep="—"),
                 width="stretch",hide_index=True,height=500)
    st.caption("Flow = change in consecutively disclosed shares × prior close. Undisclosed days are not zero-filled, and secondary-market volume is never presented as fund flow.")

    section_header("RELATIVE STRENGTH",f"Theme Relative Strength vs {rrg_benchmark}",
                   "Rebased to 100 at the start of the visible window; dotted lines show the precomputed 200-day smoothing and the benchmark remains anchored at 100.")
    rs_fig=go.Figure()
    for index,ticker in enumerate(selected_rrg_tickers):
        pair=prices[[ticker,rrg_benchmark]].dropna() if ticker in prices else pd.DataFrame()
        if pair.empty: continue
        ratio_full=pair[ticker].div(pair[rrg_benchmark])
        ratio=display_window(ratio_full,rs_horizon)
        if ratio.empty or not np.isfinite(ratio.iloc[0]) or ratio.iloc[0]==0: continue
        color=COLORS.get(ticker,palette[index%len(palette)])
        rebased=ratio.div(ratio.iloc[0]).mul(100)
        rs_fig.add_trace(go.Scatter(x=rebased.index,y=rebased,mode="lines",name=ticker,
            line=dict(color=color,width=1.6),hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>RS %{{y:.2f}}<extra></extra>"))
        smooth=display_window(ratio_full.rolling(200,min_periods=200).mean().div(ratio.iloc[0]).mul(100),rs_horizon)
        rs_fig.add_trace(go.Scatter(x=smooth.index,y=smooth,mode="lines",name=f"{ticker} 200D",showlegend=False,
            line=dict(color=color,width=.8,dash="dot"),hoverinfo="skip"))
    anchor=display_window(prices[rrg_benchmark].dropna(),rs_horizon)
    if not anchor.empty:
        rs_fig.add_trace(go.Scatter(x=anchor.index,y=np.full(len(anchor),100.0),name=f"{rrg_benchmark} ANCHOR",
            line=dict(color="#edf2f7",width=1.8,dash="dash"),hovertemplate=f"<b>{rrg_benchmark} Anchor</b><br>%{{x|%Y-%m-%d}}<br>100.00<extra></extra>"))
    base_layout(rs_fig,500,"x unified")
    rs_fig.add_hline(y=100,line_dash="dash",line_color="#728198",line_width=1)
    rs_fig.update_xaxes(title=None)
    rs_fig.update_yaxes(title="Relative Performance · Rebased 100")
    st.plotly_chart(rs_fig,width="stretch",config={"displaylogo":False})

    section_header("TACTICAL STOCK RADAR",f"Security Strength & Participation · {selected_theme}",f"Price, relative-strength averages vs {rrg_benchmark}, RVOL, dollar volume and anchored VWAP. Dollar volume is not net flow.")
    tactical=theme_tactical_scoreboard(prices,theme_ohlcv,selected_rrg_tickers,rrg_benchmark)
    tactical_sort=st.radio("Tactical sort",["Momentum high → low","RVOL high → low","Universe order"],horizontal=True,key="theme_tactical_sort")
    if tactical.empty:
        st.warning("Insufficient price or volume history to build the tactical matrix for this theme.")
    else:
        if tactical_sort=="Momentum high → low": tactical=tactical.sort_values("_MomentumScore",ascending=False)
        elif tactical_sort=="RVOL high → low": tactical=tactical.sort_values("_RVOL",ascending=False,na_position="last")
        else:
            order={ticker:index for index,ticker in enumerate(selected_rrg_tickers)}
            tactical=tactical.assign(_order=tactical["Ticker"].str.split(" · ").str[0].map(order)).sort_values("_order").drop(columns="_order")
        tactical_display=tactical.drop(columns=["_MomentumScore","_RVOL"])
        tactical_style=tactical_display.style.map(
            lambda value:"background-color:rgba(246,196,83,.16);color:#f6c453;font-weight:700" if "Volume Surge" in str(value) else "",
            subset=["RVOL (20D)"])
        st.dataframe(tactical_style,width="stretch",hide_index=True,height=min(470,42+36*len(tactical_display)))

    scoreboard=sector_rs_scoreboard(prices)
    section_header("SECTOR SCOREBOARD","Sector & Semiconductor ETF Regime Matrix","Relative to SPY across 20 / 50 / 200-day moving-average structures.")
    score_sort=st.radio("Sort regime",["Default sector order","Strong → Weak","Weak → Strong"],horizontal=True,key="scoreboard_sort")
    if not scoreboard.empty and score_sort!="Default sector order":
        regime_rank={"🔥 Absolute Strength":4,"⚡ Counter-Trend Rebound":3,"🔄 Transition":2,"⚠️ Strong Pullback":1,"❄️ Weak Breakdown":0}
        scoreboard=scoreboard.assign(_rank=scoreboard["Regime"].map(regime_rank).fillna(2)).sort_values("_rank",ascending=score_sort=="Weak → Strong").drop(columns="_rank")
    if scoreboard.empty:
        st.warning("Insufficient sector relative-strength history.")
    else:
        score_style=scoreboard.style.format({"RS vs 20EMA":"{:+.2f}%","RS vs 50EMA":"{:+.2f}%","RS vs 200SMA":"{:+.2f}%"},na_rep="—").map(
            color_return,subset=["RS vs 20EMA","RS vs 50EMA","RS vs 200SMA"])
        st.dataframe(score_style,width="stretch",hide_index=True,height=455)
        st.caption("Relative-strength regimes are descriptive diagnostics, not trading instructions.")

with tab_market:
    tape_tab, ndx_tab, matrix_tab = st.tabs(["MARKET BREADTH TAPE", "NDX LEADERS", "SECTOR MATRIX"])
    with tape_tab:
        breadth_scope = st.radio("Breadth Universe", ["S&P CORE", "NASDAQ-100"], horizontal=True,
                                 label_visibility="collapsed", key="breadth_scope")
        universe = HEAT_UNIVERSE if breadth_scope == "S&P CORE" else NASDAQ_UNIVERSE
        benchmark = "SPY" if breadth_scope == "S&P CORE" else "QQQ"
        universe_tickers = list(universe)
        breadth = breadth_history(prices, universe_tickers)
        latest_breadth = breadth.dropna(subset=["net_pct"]).iloc[-1] if not breadth.dropna(subset=["net_pct"]).empty else None

        breadth_cards = st.columns(6)
        card_values = [
            ("ADVANCERS", int(latest_breadth["adv"]) if latest_breadth is not None else None, None),
            ("DECLINERS", int(latest_breadth["dec"]) if latest_breadth is not None else None, None),
            ("NET BREADTH", float(latest_breadth["net_pct"]) if latest_breadth is not None else np.nan, "%"),
            ("ABOVE 20DMA", percent_above_ma(prices, universe_tickers, 20), "%"),
            ("ABOVE 50DMA", percent_above_ma(prices, universe_tickers, 50), "%"),
            ("ABOVE 200DMA", percent_above_ma(prices, universe_tickers, 200), "%"),
        ]
        for col, (label, value, suffix) in zip(breadth_cards, card_values):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                shown = "N/A"
            elif suffix == "%":
                shown = f"{value:.1f}%"
            else:
                shown = str(value)
            col.metric(label, shown, delta_color="off")

        left_chart, right_chart = st.columns(2, gap="medium")
        with left_chart:
            section_header("INDEX TREND", f"{benchmark} · Price & Moving Averages", "One-year price trend with 20 / 50 / 200-day moving averages.")
            s = prices[benchmark].dropna().tail(300)
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=s.index, y=s, name=benchmark, line=dict(color="#f0a13a", width=2),
                                           fill="tozeroy", fillcolor="rgba(240,161,58,.08)"))
            for window, color in [(20, "#39c6ff"), (50, "#a78bfa"), (200, "#8c99aa")]:
                ma = prices[benchmark].dropna().rolling(window, min_periods=window).mean().tail(300)
                price_fig.add_trace(go.Scatter(x=ma.index, y=ma, name=f"SMA {window}", line=dict(color=color, width=1, dash="dot")))
            base_layout(price_fig, 390, "x unified")
            price_fig.update_yaxes(title="Adjusted Close")
            st.plotly_chart(price_fig, width="stretch", config={"displaylogo": False})
        with right_chart:
            section_header("INTERNALS", f"{breadth_scope} · Net Breadth %", "(Advancers − decliners) / active sample; the orange line is the 20-day mean.")
            breadth_fig = go.Figure()
            recent = breadth.tail(300)
            breadth_fig.add_trace(go.Scatter(x=recent.index, y=recent["net_pct"], name="Daily Net Breadth",
                                             line=dict(color="#2987cf", width=1.4)))
            breadth_fig.add_trace(go.Scatter(x=recent.index, y=recent["net_pct"].rolling(20, min_periods=5).mean(),
                                             name="20D Mean", line=dict(color="#f0a13a", width=1.8)))
            base_layout(breadth_fig, 390, "x unified")
            breadth_fig.add_hline(y=0, line_dash="dash", line_color="#7a8798", line_width=1)
            breadth_fig.update_yaxes(title="Net Breadth %", range=[-105, 105])
            st.plotly_chart(breadth_fig, width="stretch", config={"displaylogo": False})

        movers = mover_frame(prices, universe)
        gain_col, loss_col = st.columns(2, gap="medium")
        for col, title, frame in [
            (gain_col, "TOP 10 GAINERS", movers.nlargest(10, "1D %") if not movers.empty else movers),
            (loss_col, "TOP 10 LOSERS", movers.nsmallest(10, "1D %") if not movers.empty else movers),
        ]:
            with col:
                section_header("LEADERSHIP TAPE", title, f"Latest-session ranking across the {breadth_scope} coverage universe.")
                styled_movers = frame.style.format({"Last": "${:,.2f}", "1D %": "{:+.2f}%"}, na_rep="—").map(
                    color_return, subset=["1D %"])
                st.dataframe(styled_movers, width="stretch", hide_index=True, height=380)

    with ndx_tab:
        ndx_movers = mover_frame(prices, NASDAQ_UNIVERSE)
        gain_col, loss_col = st.columns(2, gap="medium")
        for col, title, frame in [
            (gain_col, "TOP 10 GAINERS · NDX", ndx_movers.nlargest(10, "1D %") if not ndx_movers.empty else ndx_movers),
            (loss_col, "TOP 10 LOSERS · NDX", ndx_movers.nsmallest(10, "1D %") if not ndx_movers.empty else ndx_movers),
        ]:
            with col:
                section_header("NASDAQ LEADERSHIP", title, "Ranked by latest-session return across the dashboard's Nasdaq coverage universe.")
                styled_ndx = frame.style.format({"Last":"${:,.2f}","1D %":"{:+.2f}%"},na_rep="—").map(
                    color_return,subset=["1D %"])
                st.dataframe(styled_ndx,width="stretch",hide_index=True,height=405)
        section_header("INDEX LEADERSHIP", "Nasdaq Core Holdings · Rebased 100", "One-year normalized performance of core index constituents.")
        ndx_fig = go.Figure()
        top_holdings = sorted(NASDAQ_UNIVERSE, key=lambda t: NASDAQ_UNIVERSE[t][2], reverse=True)[:10]
        for ticker in top_holdings:
            if ticker not in prices:
                continue
            s = prices[ticker].dropna().tail(252)
            if s.empty:
                continue
            rebased = s.div(s.iloc[0]).mul(100)
            ndx_fig.add_trace(go.Scatter(x=rebased.index,y=rebased,name=ticker,mode="lines",line=dict(width=1.5),
                                         hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>"))
        base_layout(ndx_fig,360,"x unified")
        ndx_fig.add_hline(y=100,line_dash="dot",line_color="#718198",line_width=1)
        ndx_fig.update_yaxes(title="Rebased 100")
        st.plotly_chart(ndx_fig,width="stretch",config={"displaylogo":False})

    with matrix_tab:
        col_left, col_right = st.columns(2, gap="medium")
        rows = []
        for ticker, (en_name, cn_name, weight) in SECTORS.items():
            s = prices[ticker] if ticker in prices else pd.Series(dtype=float); day = last_return(s, 1)
            rows.append({"Sector":f"{en_name} · {cn_name}","ETF Code":ticker,"SPY Weight (%)":weight,"1D Return":day,
                         "1W Return":last_return(s,5),"1M Return":last_return(s,21),"YTD Return":ytd_return(s),
                         "Contribution (bp)":weight*day if np.isfinite(day) else np.nan})
        perf_df = pd.DataFrame(rows)
        with col_left:
            section_header("SECTOR ATTRIBUTION", "Sector Performance & Contribution", "SPY sector proxy weights, multi-horizon returns and one-day contribution. Weights are static approximations.")
            styled = perf_df.style.format({"SPY Weight (%)":"{:.1f}","1D Return":"{:+.2f}%","1W Return":"{:+.2f}%","1M Return":"{:+.2f}%","YTD Return":"{:+.2f}%","Contribution (bp)":"{:+.2f}"}, na_rep="—").map(color_return, subset=["1D Return","1W Return","1M Return","YTD Return","Contribution (bp)"])
            st.dataframe(styled, width="stretch", hide_index=True, height=510)
        with col_right:
            section_header("LONG-TERM REGIME", "200-Day MA Slope", "Regression slope of the 200-day moving average over the latest 20 sessions, normalized by the current SMA200.")
            slope_rows=[]
            for ticker,(en_name,cn_name,_) in SECTORS.items():
                slope=normalized_slope(prices[ticker]) if ticker in prices else np.nan
                slope_rows.append({"Sector":f"{en_name} · {cn_name}","ETF":ticker,"Slope % / Day":slope,"Trend State":trend_label(slope)})
            slope_df=pd.DataFrame(slope_rows)
            st.dataframe(slope_df.style.format({"Slope % / Day":"{:+.4f}%"},na_rep="—").map(color_return,subset=["Slope % / Day"]),width="stretch",hide_index=True,height=260)
            fig=go.Figure()
            for ticker,(_,cn_name,_) in SECTORS.items():
                s=prices[ticker].dropna().tail(252) if ticker in prices else pd.Series(dtype=float)
                if s.empty: continue
                rebased=s.div(s.iloc[0]).mul(100)
                fig.add_trace(go.Scatter(x=rebased.index,y=rebased,mode="lines",name=ticker,line=dict(color=COLORS[ticker],width=1.25),hovertemplate=f"<b>{ticker} · {cn_name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>"))
            base_layout(fig,310,"x unified"); fig.add_hline(y=100,line_dash="dot",line_color="#718198",line_width=1)
            fig.update_layout(margin=dict(l=45,r=14,t=28,b=35)); fig.update_yaxes(title="1Y Rebased 100")
            st.plotly_chart(fig,width="stretch",config={"displaylogo":False})

with tab_heat:
    heat_control_0,heat_control_1,heat_control_2,_=st.columns([1.35,1.2,1,3.5])
    with heat_control_0:
        heat_mode=st.radio("Heat Map Engine",["TradingView Official","Quant Native"],horizontal=True,
                           label_visibility="collapsed",key="heat_mode")
    with heat_control_1:
        heat_index = st.radio("Heat Map Universe", ["S&P 500 CORE", "NASDAQ-100"], horizontal=True,
                              label_visibility="collapsed", key="heat_index")
    with heat_control_2:
        heat_period = st.radio("Return Window", ["1D", "1M"], horizontal=True,
                               label_visibility="collapsed", key="heat_period")
    heat_universe = HEAT_UNIVERSE if heat_index == "S&P 500 CORE" else NASDAQ_UNIVERSE
    section_header("STOCK HEATMAP",f"{heat_index} · {heat_period} Change",
                   "Official mode embeds the TradingView Stock Heatmap; Quant Native uses yfinance and Plotly.")
    if heat_mode == "TradingView Official":
        tv_source="SPX500" if heat_index=="S&P 500 CORE" else "NASDAQ100"
        tv_color="change" if heat_period=="1D" else "Perf.1M"
        st.html(f'''<style>html,body{{height:100%;margin:0;overflow:hidden;background:#000}}.tradingview-widget-container{{height:730px!important;width:100%!important}}.tradingview-widget-container__widget{{height:700px!important;width:100%!important}}</style>
        <div class="tradingview-widget-container" style="height:730px;width:100%;background:#000">
        <div class="tradingview-widget-container__widget" style="height:700px;width:100%"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/heatmap/stock/" rel="noopener nofollow" target="_blank"><span style="color:#2962ff">Stock Heatmap</span></a> by TradingView</div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
        {{"dataSource":"{tv_source}","blockSize":"market_cap_basic","blockColor":"{tv_color}","grouping":"sector","locale":"en","colorTheme":"dark","hasTopBar":true,"isDataSetEnabled":true,"isZoomEnabled":true,"hasSymbolTooltip":true,"isMonoSize":false,"width":"100%","height":700}}
        </script></div>''',width="stretch",unsafe_allow_javascript=True)
        st.caption("The official widget is priced and rendered directly by TradingView. Switch to Quant Native if third-party scripts are blocked.")
    else:
        rows=[]; periods=1 if heat_period=="1D" else 21
        for ticker,meta in heat_universe.items():
            if ticker not in prices: continue
            ret=last_return(prices[ticker],periods)
            if not np.isfinite(ret): continue
            if heat_index=="S&P 500 CORE": company,sector_etf,weight=meta; group=f"{sector_etf} · {SECTORS[sector_etf][1]}"
            else: company,group,weight=meta
            rows.append({"Group":group,"Ticker":ticker,"Company":company,"Market Weight":weight,"Return":ret,"Tile Text":f"{ret:+.2f}%"})
        heat_df=pd.DataFrame(rows)
        if heat_df.empty: st.warning("No valid observations are available for the heat-map universe.")
        else:
            floor=2.0 if heat_period=="1D" else 5.0; limit=max(floor,float(heat_df["Return"].abs().quantile(.92)))
            fig=px.treemap(heat_df,path=[px.Constant(heat_index),"Group","Ticker"],values="Market Weight",color="Return",
                color_continuous_scale=[(0,"#b51f32"),(.32,"#5c1d29"),(.5,"#34373c"),(.68,"#125338"),(1,"#00a85a")],
                range_color=(-limit,limit),custom_data=["Company","Return","Market Weight","Tile Text"])
            tile_text=[str(custom[3]) if str(node_id).count("/")>=2 else "" for node_id,custom in zip(fig.data[0].ids,fig.data[0].customdata)]
            fig.update_traces(text=tile_text,texttemplate="<b>%{label}</b><br>%{text}",hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>Return %{customdata[1]:+.2f}%<br>Weight proxy %{customdata[2]:.2f}%<extra></extra>",textfont=dict(size=15,color="#f7f8fa"),marker=dict(line=dict(color="#05070a",width=2.2)),root_color="#0a0d14",tiling=dict(packing="squarify",pad=2))
            base_layout(fig,720)
            fig.update_layout(margin=dict(l=2,r=2,t=10,b=2),coloraxis_colorbar=dict(title=f"{heat_period} %",thickness=11,len=.58,tickformat="+.1f"))
            st.plotly_chart(fig,width="stretch",config={"displaylogo":False})

    section_header("TRADING PRESSURE",f"{heat_index} · 20D Secondary-Market Activity","Measures activity on up and down days only. It does not represent ETF creations, redemptions or primary-market flow.")
    try:
        flow_close,flow_volume=load_flow_data(tuple(heat_universe),"3mo")
    except Exception:
        flow_close,flow_volume=pd.DataFrame(),pd.DataFrame()
    flow_rows=[]
    for ticker,meta in heat_universe.items():
        if ticker not in flow_close or ticker not in flow_volume: continue
        score=signed_flow_score(flow_close[ticker],flow_volume[ticker])
        if not np.isfinite(score): continue
        if heat_index=="S&P 500 CORE": company,sector_etf,weight=meta; group=f"{sector_etf} · {SECTORS[sector_etf][1]}"
        else: company,group,weight=meta
        flow_rows.append({"Group":group,"Ticker":ticker,"Company":company,"Market Weight":weight,"Flow":score,"Flow Text":f"{score:+.1f}%"})
    flow_df=pd.DataFrame(flow_rows)
    if flow_df.empty: st.warning("No valid secondary-market activity observations are available.")
    else:
        flow_limit=max(20.0,float(flow_df["Flow"].abs().quantile(.92)))
        flow_fig=px.treemap(flow_df,path=[px.Constant(f"{heat_index} ACTIVITY"),"Group","Ticker"],values="Market Weight",color="Flow",
            color_continuous_scale=[(0,"#a61b3b"),(.5,"#2e333b"),(1,"#087f60")],range_color=(-flow_limit,flow_limit),
            custom_data=["Company","Flow","Flow Text"])
        flow_text=[str(custom[2]) if str(node_id).count("/")>=2 else "" for node_id,custom in zip(flow_fig.data[0].ids,flow_fig.data[0].customdata)]
        flow_fig.update_traces(text=flow_text,texttemplate="<b>%{label}</b><br>%{text}",hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>20D Trading Pressure %{customdata[1]:+.1f}%<extra></extra>",marker=dict(line=dict(color="#05070a",width=2)),root_color="#0a0d14",tiling=dict(pad=2))
        base_layout(flow_fig,540)
        flow_fig.update_layout(margin=dict(l=2,r=2,t=8,b=2),coloraxis_colorbar=dict(title="Activity %",thickness=11,len=.58))
        st.plotly_chart(flow_fig,width="stretch",config={"displaylogo":False})

with tab_macro:
    col_left,col_right=st.columns(2,gap="medium")
    with col_left:
        section_header("CROSS-ASSET TAPE","Commodities Monitor","Multi-horizon performance across precious metals, energy, the U.S. dollar and duration assets.")
        rows=[]
        for ticker,name in COMMODITIES.items():
            s=prices[ticker] if ticker in prices else pd.Series(dtype=float); clean=s.dropna()
            rows.append({"Asset":name,"Ticker":ticker,"Last":float(clean.iloc[-1]) if not clean.empty else np.nan,"1D":last_return(s,1),"1W":last_return(s,5),"1M":last_return(s,21),"3M":last_return(s,63),"YTD":ytd_return(s)})
        df=pd.DataFrame(rows)
        styled=df.style.format({"Last":"${:,.2f}","1D":"{:+.2f}%","1W":"{:+.2f}%","1M":"{:+.2f}%","3M":"{:+.2f}%","YTD":"{:+.2f}%"},na_rep="—").map(color_return,subset=["1D","1W","1M","3M","YTD"])
        st.dataframe(styled,width="stretch",hide_index=True,height=510)
    with col_right:
        section_header("MEAN REVERSION RADAR","Rolling Cross-Asset Z-Scores","Rolling standardized ratios of GLD, TLT and QQQ versus SPY; ±2σ marks statistical extremes.")
        fig=go.Figure()
        for ticker,name in {"GLD":"Gold","TLT":"Long Duration","QQQ":"Technology"}.items():
            if ticker not in prices: continue
            z=rolling_zscore(prices[ticker],prices["SPY"],z_window).tail(252)
            if z.empty: continue
            fig.add_trace(go.Scatter(x=z.index,y=z,mode="lines",name=f"{ticker} · {name}",line=dict(color=COLORS[ticker],width=1.8),hovertemplate=f"<b>{ticker} · {name}</b><br>%{{x|%Y-%m-%d}}<br>Z %{{y:.2f}}σ<extra></extra>"))
        base_layout(fig,570,"x unified")
        fig.add_hline(y=2,line_dash="dash",line_color="#ff4f70",line_width=1.2,annotation_text="+2σ OVERBOUGHT",annotation_font_color="#ff6a84")
        fig.add_hline(y=-2,line_dash="dash",line_color="#18d39c",line_width=1.2,annotation_text="−2σ OVERSOLD",annotation_font_color="#2be0ac")
        fig.add_hline(y=0,line_dash="dot",line_color="#718198",line_width=1)
        fig.update_xaxes(title=None); fig.update_yaxes(title="Rolling Z-Score (σ)")
        st.plotly_chart(fig,width="stretch",config={"displaylogo":False})

with tab_technical:
    section_header("TECHNICALS", "Relative Strength · MA Regime · Z-Score", "The blue line is the relative-strength ratio. Below both averages indicates a watch zone; above both uses Z-score to distinguish hold from overheat.")
    control_1, control_2, control_3, control_4 = st.columns([1.2,1,1,1])
    with control_1:
        preset = st.selectbox("Preset", ["CUSTOM","XLE / SPY","XLK / SPY","SMH / SPY","GDX / GLD","TLT / SPY"], key="rs_preset")
    preset_pair = preset.replace(" ", "").split("/") if preset != "CUSTOM" else None
    with control_2:
        numerator_input = st.text_input("Numerator", value=preset_pair[0] if preset_pair else "XLE", key=f"rs_num_{preset}")
    with control_3:
        denominator_input = st.text_input("Denominator", value=preset_pair[1] if preset_pair else "SPY", key=f"rs_den_{preset}")
    with control_4:
        technical_period = st.selectbox("History",list(PERIOD_BARS),index=4,key="rs_period")
    numerator = "".join(c for c in numerator_input.upper().strip().replace(".", "-") if c.isalnum() or c in "-^=")[:15]
    denominator = "".join(c for c in denominator_input.upper().strip().replace(".", "-") if c.isalnum() or c in "-^=")[:15]

    ratio_data = pd.DataFrame()
    if numerator and denominator:
        try:
            requested=list(dict.fromkeys([numerator,denominator,"SPY","QQQ"]))
            ratio_data=load_custom_prices(tuple(requested),"max")
        except Exception as exc:
            st.warning(f"Relative-strength download failed: {exc}")
    pair_ok = (numerator in ratio_data and denominator in ratio_data and
               not ratio_data[numerator].dropna().empty and not ratio_data[denominator].dropna().empty)
    if not pair_ok:
        st.warning("Enter numerator and denominator symbols recognized by Yahoo Finance.")
    else:
        ratio_frame, ratio_stats = relative_signal(ratio_data[numerator],ratio_data[denominator])
        if ratio_frame.empty:
            st.warning("Fewer than 200 common sessions; the 60/200-day regime cannot be calculated.")
        else:
            stat_cols = st.columns(5)
            stat_cols[0].metric("REGIME",str(ratio_stats["state"]),delta_color="off")
            stat_cols[1].metric("RATIO",f"{ratio_stats['ratio']:.5f}",delta_color="off")
            stat_cols[2].metric("Z-SCORE",f"{ratio_stats['zscore']:+.2f}σ",delta_color="off")
            stat_cols[3].metric("VS 60MA",f"{ratio_stats['vs_ma60']:+.2f}%")
            stat_cols[4].metric("VS 200MA",f"{ratio_stats['vs_ma200']:+.2f}%")

            active_ratio=display_window(ratio_frame,technical_period)
            plot_ratio=display_with_preroll(ratio_frame,technical_period)
            technical_fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,row_heights=[.68,.32])
            technical_fig.add_trace(go.Scatter(x=plot_ratio.index,y=plot_ratio["ratio"],name="Relative Ratio",
                line=dict(color="#4f86d9",width=2)),row=1,col=1)
            technical_fig.add_trace(go.Scatter(x=plot_ratio.index,y=plot_ratio["ma60"],name="60MA",
                line=dict(color="#e56f24",width=1.7)),row=1,col=1)
            technical_fig.add_trace(go.Scatter(x=plot_ratio.index,y=plot_ratio["ma200"],name="200MA",
                line=dict(color="#a8adb5",width=1.6)),row=1,col=1)
            hist_colors=np.where(plot_ratio["hist"]>=0,"#29a987","#e74b5e")
            technical_fig.add_trace(go.Bar(x=plot_ratio.index,y=plot_ratio["hist"],name="Histogram",
                marker_color=hist_colors,opacity=.8),row=2,col=1)
            technical_fig.add_trace(go.Scatter(x=plot_ratio.index,y=plot_ratio["macd"],name="MACD 8/17",
                line=dict(color="#4f86d9",width=1.4)),row=2,col=1)
            technical_fig.add_trace(go.Scatter(x=plot_ratio.index,y=plot_ratio["signal"],name="Signal 9",
                line=dict(color="#e56f24",width=1.3)),row=2,col=1)
            base_layout(technical_fig,620,"x unified")
            technical_fig.update_layout(title=dict(text=f"{numerator} / {denominator} · Ratio + 60/200MA · MACD (8,17,9)",font=dict(size=12,color="#f0a13a")))
            technical_fig.update_yaxes(title="Relative Ratio",row=1,col=1)
            technical_fig.update_yaxes(title="MACD",row=2,col=1,zeroline=True,zerolinecolor="#697789")
            if not active_ratio.empty and active_ratio.index[0]>plot_ratio.index[0]:
                technical_fig.add_vline(x=active_ratio.index[0].to_pydatetime(),line_dash="dot",line_color="#74849a",
                    annotation_text=f"{technical_period} ACTIVE WINDOW",annotation_position="top left")
            st.plotly_chart(technical_fig,width="stretch",config={"displaylogo":False})

            anchor_col,readout_col=st.columns([1.25,.75],gap="medium")
            with anchor_col:
                section_header("ANCHOR COMPARISON", "SPY / QQQ Anchored Performance", "Numerator, denominator, SPY and QQQ are rebased to 100 at the start of the selected valid window.")
                anchor_fig=go.Figure(); anchor_colors={numerator:"#4f86d9",denominator:"#e56f24","SPY":"#e8edf4","QQQ":"#a78bfa"}
                for ticker in dict.fromkeys([numerator,denominator,"SPY","QQQ"]):
                    if ticker not in ratio_data: continue
                    series=display_window(ratio_data[ticker].dropna(),technical_period)
                    if series.empty: continue
                    rebased=series.div(series.iloc[0]).mul(100)
                    anchor_fig.add_trace(go.Scatter(x=rebased.index,y=rebased,name=ticker,line=dict(color=anchor_colors.get(ticker,"#8ca0b8"),width=1.8),
                        hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>"))
                base_layout(anchor_fig,360,"x unified"); anchor_fig.add_hline(y=100,line_dash="dot",line_color="#718198",line_width=1)
                anchor_fig.update_yaxes(title="Anchored 100")
                st.plotly_chart(anchor_fig,width="stretch",config={"displaylogo":False})
            with readout_col:
                sentiment,drivers,sentiment_score=market_sentiment(prices)
                section_header("RISK APPETITE", "Cross-Asset Market State", "Classified from the relative trends of technology, small caps and credit versus gold, duration and volatility.")
                state_color="#31d6a0" if sentiment_score>=3 else ("#ff5874" if sentiment_score<=-3 else "#e8bd55")
                st.markdown(f"""<div style="border:1px solid #24354d;background:#0a111c;padding:16px;border-radius:4px;line-height:1.7;color:#aebdd0;min-height:285px">
                <div style="font-size:1.15rem;font-weight:750;color:{state_color}">{sentiment}</div>
                <div style="margin-top:8px">Current {numerator}/{denominator}: <b>{ratio_stats['state']}</b>; vs 60MA {ratio_stats['vs_ma60']:+.2f}%, vs 200MA {ratio_stats['vs_ma200']:+.2f}%.</div>
                <div style="margin-top:8px;color:#8191a6">Drivers: {drivers or 'Insufficient data'}</div>
                <div style="margin-top:8px;color:#65768d">Score {sentiment_score:+d}; positive favors risk appetite, negative favors defense. Anchored curves support comparison and are not valuation measures.</div></div>""",unsafe_allow_html=True)

    st.divider()
    slope_head,slope_control,slope_range=st.columns([4.2,1.25,1.0],vertical_alignment="bottom")
    with slope_head:
        section_header("SMA SLOPES","Moving-Average Slope Momentum",
            "Calculated on full history: slope = (MA today − MA five sessions ago) / (5 × prior MA) × 100%, in percent per trading day.")
    with slope_control:
        slope_choice=st.selectbox("Slope Ticker",["QQQ","SPY","SMH","SOXX","IWM","Custom"],key="sma_slope_ticker")
    with slope_range:
        slope_period=st.selectbox("Chart Range",["6M","1Y","3Y","5Y","MAX"],index=2,key="sma_slope_range")
    slope_ticker=slope_choice
    if slope_choice=="Custom":
        slope_ticker="".join(ch for ch in st.text_input("Custom slope ticker",value="KWEB",key="sma_slope_custom").upper().strip().replace(".","-")
                             if ch.isalnum() or ch in "-^=")[:15]
    slope_close=prices[slope_ticker].dropna() if slope_ticker in prices else pd.Series(dtype=float)
    if slope_ticker:
        try:
            slope_download=(ratio_data if slope_ticker in ratio_data else
                            load_custom_prices((slope_ticker,),"max"))
            if slope_ticker in slope_download and not slope_download[slope_ticker].dropna().empty:
                slope_close=slope_download[slope_ticker].dropna().combine_first(slope_close).sort_index()
        except Exception:
            pass
    if len(slope_close)<205:
        st.info(f"{slope_ticker or 'Symbol'} has fewer than 205 valid closes; the five-day slope of the 200DMA is unavailable.")
    else:
        slope_panel=cached_sma_slope_panel(slope_close)
        slope_stats=slope_statistics(slope_panel)
        slope_state=slope_regime(slope_panel)
        if slope_state=="BULLISH_DIP":
            st.success("🟢 BULLISH DIP: the long-term average is still rising while the short-term slope pulls back. Watchlist condition, not a guaranteed entry.")
        elif slope_state=="SYSTEMIC_BREAK":
            st.error("🚨 SYSTEMIC TREND BREAK: 20/50/200DMA slopes are all negative. Research risk flag: suspend new leveraged-long exposure.")
        elif slope_state=="INSUFFICIENT":
            st.info("All three moving-average slopes have not formed yet.")
        else:
            st.info("⚪ Mixed slope regime: neither the bullish-dip nor all-negative rule is active.")
        slope_plot=display_window(slope_panel,slope_period)
        slope_fig=go.Figure()
        for window,color in ((20,"#4f86d9"),(50,"#e56f24"),(200,"#ff5874")):
            column=f"SMA{window} Slope"
            slope_fig.add_trace(go.Scatter(x=slope_plot.index,y=slope_plot[column],name=f"{window}DMA Slope",
                mode="lines",line=dict(color=color,width=2 if window==20 else 1.7),
                hovertemplate=f"<b>{window}DMA</b><br>%{{x|%Y-%m-%d}}<br>%{{y:+.4f}}%/day<extra></extra>"))
        base_layout(slope_fig,420,"x unified")
        slope_fig.add_hline(y=0,line_dash="dash",line_color="#f3f5f9",line_width=1.3)
        slope_fig.update_yaxes(title="Slope · % / trading day",ticksuffix="%")
        slope_fig.update_xaxes(title=None)
        st.plotly_chart(slope_fig,width="stretch",config={"displaylogo":False,"scrollZoom":True})
        st.dataframe(slope_stats.style.format({"Current":"{:+.4f}%","Mean":"{:+.4f}%",
            "Std":"{:.4f}%","Min":"{:+.4f}%","Max":"{:+.4f}%",
            "% days negative":"{:.1f}%"},na_rep="—").map(color_return,subset=["Current","Mean","Min","Max"]),
            width="stretch",hide_index=True,height=155)
        st.caption(f"Distribution statistics use all valid {slope_ticker} history, not only the visible window, through {slope_panel.index[-1]:%Y-%m-%d}. The regime is a research alert, not a crash forecast or trading control.")

    st.divider()
    beta_head,beta_control=st.columns([4.2,1.15],vertical_alignment="bottom")
    with beta_head:
        section_header("BETA ATTRIBUTION","CAPM Return Decomposition",
            "Uses SPY as the single market factor. Beta is estimated from 252 common sessions; the latest return is decomposed into Beta × SPY return and the remaining residual.")
    with beta_control:
        beta_choice=st.selectbox("Target Ticker",["SMH","KWEB","BABA","SOXX","QQQ","NVDA","Custom"],key="beta_target")
    beta_ticker=beta_choice
    if beta_choice=="Custom":
        beta_ticker="".join(ch for ch in st.text_input("Custom attribution ticker",value="TSM",key="beta_custom").upper().strip().replace(".","-")
                             if ch.isalnum() or ch in "-^=")[:15]
    beta_target=prices[beta_ticker].dropna() if beta_ticker in prices else pd.Series(dtype=float)
    if beta_ticker and len(beta_target)<253:
        try:
            beta_download=load_custom_prices((beta_ticker,),"max")
            beta_target=beta_download[beta_ticker].dropna() if beta_ticker in beta_download else pd.Series(dtype=float)
        except Exception:
            beta_target=pd.Series(dtype=float)
    beta_result=(cached_beta_decomposition(beta_target,prices["SPY"].dropna())
                 if beta_ticker and beta_ticker!="SPY" and not beta_target.empty else None)
    if beta_ticker=="SPY":
        st.info("Target and benchmark are both SPY. Select another symbol to inspect the specific residual.")
    elif beta_result is None:
        st.info(f"{beta_ticker or 'Target'} does not have 252 valid common returns with SPY, or benchmark volatility is zero.")
    else:
        latest_spy_session=prices["SPY"].dropna().index[-1]
        if beta_result["asof"]<latest_spy_session:
            st.warning(f"The latest common close for {beta_ticker} is {beta_result['asof']:%Y-%m-%d}, behind SPY. The attribution is for that date, not today.")
        beta_cards=st.columns(4)
        beta_cards[0].metric("TARGET 1D",f"{beta_result['target_1d']:+.2f}%")
        beta_cards[1].metric("SPY 1D",f"{beta_result['market_1d']:+.2f}%")
        beta_cards[2].metric("252D BETA",f"{beta_result['beta']:+.2f}",f"ρ {beta_result['correlation']:+.2f}",delta_color="off")
        beta_cards[3].metric("SPECIFIC RESIDUAL",f"{beta_result['specific_pp']:+.2f} pp")
        attribution_fig=go.Figure()
        for name,value,color in (("Explained by SPY × Beta",beta_result["explained_pp"],"#4f86d9"),
                                 ("Specific Remainder",beta_result["specific_pp"],"#f0a13a")):
            attribution_fig.add_trace(go.Bar(x=[value],y=["Latest Return"],orientation="h",name=name,
                marker_color=color,text=[f"{value:+.2f} pp"],textposition="auto",
                hovertemplate=f"<b>{name}</b><br>%{{x:+.3f}} percentage points<extra></extra>"))
        attribution_fig.add_trace(go.Scatter(x=[beta_result["target_1d"]],y=["Latest Return"],mode="markers",
            name=f"Target total {beta_result['target_1d']:+.2f}%",marker=dict(color="#f4f6fb",size=13,symbol="diamond"),
            hovertemplate="Target total %{x:+.3f}%<extra></extra>"))
        base_layout(attribution_fig,295,"closest")
        attribution_fig.update_layout(barmode="relative",title=dict(text=f"{beta_ticker} · {beta_result['asof']:%Y-%m-%d} · 1D Market vs Specific",font=dict(size=12,color="#f0a13a")))
        attribution_fig.add_vline(x=0,line_dash="dash",line_color="#f3f5f9",line_width=1)
        attribution_fig.update_xaxes(title="Return contribution · percentage points",ticksuffix=" pp")
        attribution_fig.update_yaxes(title=None)
        st.plotly_chart(attribution_fig,width="stretch",config={"displaylogo":False})
        st.caption(f"Identity: {beta_result['explained_pp']:+.2f} pp market contribution + {beta_result['specific_pp']:+.2f} pp residual = {beta_result['target_1d']:+.2f}% for {beta_ticker}. The residual includes unmodeled industry, currency and security-specific effects; it is not risk-adjusted CAPM alpha or causal proof.")

    section_header("REGIME SCANNER", "Bottoming & Overheat Scanner", "Below the 60/200-day averages enters the watch zone; Z≤−1.5 with a rising MACD histogram flags a bottoming candidate; above both is hold, and Z≥+2 is overheated.")
    scan_candidates = list(dict.fromkeys(list(SECTORS) + ["SMH","GLD","SLV","TLT","USO","CPER","QQQ","IWM"]
                                     + sorted(NASDAQ_UNIVERSE,key=lambda t:NASDAQ_UNIVERSE[t][2],reverse=True)[:15]))
    scan_rows=[]
    for ticker in scan_candidates:
        if ticker not in prices or ticker == "SPY":
            continue
        _, stats = relative_signal(prices[ticker],prices["SPY"])
        if stats.get("state") == "Insufficient data":
            continue
        scan_rows.append({"Ticker":ticker,"State":stats["state"],"Z-Score":stats["zscore"],
                          "vs 60MA %":stats["vs_ma60"],"vs 200MA %":stats["vs_ma200"],"MACD Hist":stats["macd_hist"]})
    scan_df=pd.DataFrame(scan_rows)
    bottom_col,hot_col=st.columns(2,gap="medium")
    with bottom_col:
        section_header("BOTTOM WATCH", "Early Watch / Bottoming Candidates", "Relative strength is below both averages, ranked from lowest to highest Z-score.")
        bottom=scan_df[scan_df["State"].astype(str).str.contains("Early watch")].sort_values("Z-Score").head(12) if not scan_df.empty else scan_df
        st.dataframe(bottom.style.format({"Z-Score":"{:+.2f}","vs 60MA %":"{:+.2f}%","vs 200MA %":"{:+.2f}%","MACD Hist":"{:+.5f}"},na_rep="—").map(color_return,subset=["Z-Score","vs 60MA %","vs 200MA %"]),width="stretch",hide_index=True,height=320)
    with hot_col:
        section_header("HOLD / TRIM", "Strong Hold / Overheat Trim", "Relative strength is above both averages, with the highest Z-scores shown first.")
        hot=scan_df[scan_df["State"].astype(str).str.contains("Hold|Overheated",regex=True)].sort_values("Z-Score",ascending=False).head(12) if not scan_df.empty else scan_df
        st.dataframe(hot.style.format({"Z-Score":"{:+.2f}","vs 60MA %":"{:+.2f}%","vs 200MA %":"{:+.2f}%","MACD Hist":"{:+.5f}"},na_rep="—").map(color_return,subset=["Z-Score","vs 60MA %","vs 200MA %"]),width="stretch",hide_index=True,height=320)

    st.divider()
    section_header("SINGLE-TICKER TECHNICALS", "Single-Security Technical Analysis", "Enter any Yahoo Finance symbol to render price, 20/50/200-day moving averages and rolling Z-score.")
    single_col_1,single_col_2,_=st.columns([1,1,4])
    with single_col_1:
        single_input=st.text_input("Ticker",value="QQQ",key="single_ticker")
    with single_col_2:
        single_period=st.selectbox("History",list(PERIOD_BARS),index=4,key="single_period")
    single_ticker="".join(c for c in single_input.upper().strip().replace(".","-") if c.isalnum() or c in "-^=")[:15]
    single_data=pd.DataFrame()
    if single_ticker:
        try:
            if single_ticker in ratio_data and not ratio_data[single_ticker].dropna().empty:
                single_data=ratio_data[[single_ticker]].copy()
            else:
                single_data=load_custom_prices((single_ticker,),"max")
        except Exception as exc:
            st.warning(f"Single-symbol download failed: {exc}")
    if single_ticker not in single_data or single_data[single_ticker].dropna().empty:
        st.warning("No valid price history was returned for this symbol.")
    else:
        s=single_data[single_ticker].dropna()
        sma20=s.rolling(20,min_periods=20).mean(); sma50=s.rolling(50,min_periods=50).mean(); sma200=s.rolling(200,min_periods=200).mean()
        mean120=s.rolling(120,min_periods=120).mean(); std120=s.rolling(120,min_periods=120).std(ddof=0).replace(0,np.nan)
        price_z=s.sub(mean120).div(std120)
        latest_price=float(s.iloc[-1]); latest_z=float(price_z.dropna().iloc[-1]) if not price_z.dropna().empty else np.nan
        dist200=float((latest_price/sma200.dropna().iloc[-1]-1)*100) if not sma200.dropna().empty else np.nan
        slope=normalized_slope(s)
        single_stats=st.columns(4)
        single_stats[0].metric("PRICE",f"${latest_price:,.2f}")
        single_stats[1].metric("VS 200MA",format_num(dist200))
        single_stats[2].metric("PRICE Z-SCORE",format_num(latest_z,"σ"))
        single_stats[3].metric("200MA SLOPE",format_num(slope,"% / day"))

        continuation={"detected":False,"reason":"OHLCV history is currently unavailable."}
        try:
            continuation_panel=load_ohlcv((single_ticker,),"1y")
            if not continuation_panel.empty and single_ticker in continuation_panel.columns.get_level_values(0):
                continuation=continuation_price_diagnostics(continuation_panel[single_ticker])
        except Exception as exc:
            continuation={"detected":False,"reason":f"Continuation diagnostics failed: {exc}"}

        if continuation.get("detected"):
            continuation_score=sum([
                bool(continuation.get("avwap_pass")),bool(continuation.get("dryup_pass")),
                bool(continuation.get("ema_pass")),
            ])
            if not continuation.get("avwap_pass"):
                continuation_badge="❄️ FAILED BREAKOUT · BELOW INSTITUTIONAL COST"; continuation_class="continuation-red"
            elif continuation_score>=3:
                continuation_badge="🔥 HIGH-CONVICTION CONTINUATION SETUP"; continuation_class="continuation-gold"
            else:
                continuation_badge="🔎 POST-BREAKOUT CONSOLIDATION · MONITOR"; continuation_class="continuation-watch"
            dryup_text=f"{continuation['dryup']:.2f}×" if np.isfinite(continuation.get("dryup",np.nan)) else "Awaiting sample"
            ema_text=f"{continuation['ema_distance']:.2f}%" if np.isfinite(continuation.get("ema_distance",np.nan)) else "N/A"
            avwap_gap=(continuation["price"]/continuation["avwap"]-1)*100 if continuation.get("avwap") else np.nan
            st.markdown(f"""
            <div class="continuation-panel">
              <div class="continuation-head">
                <div><span class="continuation-title">CONTINUATION FILTER · BREAKOUT QUALITY</span><br><span class="continuation-badge {continuation_class}">{continuation_badge}</span></div>
                <div class="continuation-meta">SCORE {continuation_score}/3 · ANCHOR {continuation['anchor']:%Y-%m-%d} · +{continuation['breakout_return']:.2f}% · RVOL {continuation['breakout_rvol']:.2f}× · {continuation['days_since']} BARS AGO</div>
              </div>
              <div class="continuation-grid">
                <div class="continuation-item"><div class="continuation-label">01 · ANCHORED VWAP</div><div class="continuation-value">${continuation['avwap']:,.2f} · {avwap_gap:+.2f}%</div><div class="continuation-note">{continuation['avwap_label']}</div></div>
                <div class="continuation-item"><div class="continuation-label">02 · VOLUME DRY-UP</div><div class="continuation-value">{dryup_text}</div><div class="continuation-note">{continuation['dryup_label']}</div></div>
                <div class="continuation-item"><div class="continuation-label">03 · EMA20 CONVERGENCE</div><div class="continuation-value">{ema_text}</div><div class="continuation-note">{continuation['ema_label']}</div></div>
              </div>
            </div>
            """,unsafe_allow_html=True)
        else:
            st.info(f"CONTINUATION FILTER · {continuation.get('reason','No qualifying high-volume breakout in the lookback window.')}")

        single_full=pd.DataFrame({"price":s,"sma20":sma20,"sma50":sma50,"sma200":sma200,"zscore":price_z})
        single_full["avwap"]=np.nan
        if continuation.get("detected"):
            single_full["avwap"]=continuation["avwap_series"].reindex(single_full.index)
        single_active=display_window(single_full,single_period)
        single_plot=display_with_preroll(single_full,single_period)
        single_fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.07,row_heights=[.72,.28])
        single_fig.add_trace(go.Scatter(x=single_plot.index,y=single_plot["price"],name=single_ticker,line=dict(color="#4f86d9",width=2)),row=1,col=1)
        for name,column,color in [("SMA20","sma20","#31d6a0"),("SMA50","sma50","#e56f24"),("SMA200","sma200","#a8adb5")]:
            single_fig.add_trace(go.Scatter(x=single_plot.index,y=single_plot[column],name=name,line=dict(color=color,width=1.4)),row=1,col=1)
        if continuation.get("detected"):
            single_fig.add_trace(go.Scatter(x=single_plot.index,y=single_plot["avwap"],name="Anchored VWAP",
                line=dict(color="#f4c34f",width=2.2,dash="dash")),row=1,col=1)
        single_fig.add_trace(go.Scatter(x=single_plot.index,y=single_plot["zscore"],name="Z-Score",line=dict(color="#a78bfa",width=1.7)),row=2,col=1)
        base_layout(single_fig,620,"x unified")
        single_fig.add_hline(y=2,line_dash="dash",line_color="#ff4f70",row=2,col=1)
        single_fig.add_hline(y=-2,line_dash="dash",line_color="#18d39c",row=2,col=1)
        single_fig.add_hline(y=0,line_dash="dot",line_color="#718198",row=2,col=1)
        if not single_active.empty and single_active.index[0]>single_plot.index[0]:
            single_fig.add_vline(x=single_active.index[0].to_pydatetime(),line_dash="dot",line_color="#74849a",
                annotation_text=f"{single_period} ACTIVE WINDOW",annotation_position="top left")
        if continuation.get("detected") and continuation["anchor"]>=single_plot.index.min():
            single_fig.add_vline(x=continuation["anchor"].to_pydatetime(),line_dash="dot",line_color="#f4c34f",
                annotation_text="VOLUME BREAKOUT",annotation_position="top right",row=1,col=1)
        single_fig.update_layout(title=dict(text=f"{single_ticker} · Price / Moving Averages / Z-Score",font=dict(size=12,color="#f0a13a")))
        single_fig.update_yaxes(title="Price",row=1,col=1); single_fig.update_yaxes(title="Z",row=2,col=1)
        st.plotly_chart(single_fig,width="stretch",config={"displaylogo":False})

        st.divider()
        section_header("MULTI-TIMEFRAME TECHNICALS","Multi-Timeframe Indicator Workbench",
            "Bollinger Bands / anchored VWAP / SSL Hybrid / HA Trend / ADX-DI. Click legend items to hide; double-click to isolate a layer.")
        tf_col,bars_col=st.columns([1,1])
        with tf_col:
            technical_tf=st.selectbox("Candlestick timeframe",list(TIMEFRAME_SOURCE),index=6,key="technical_candle_tf",
                format_func=lambda value:{"1MO":"Monthly","1Y":"Yearly"}.get(value,value))
        with bars_col:
            chart_bars=st.selectbox("Visible bars",[120,240,480,"All"],index=1,key="technical_chart_bars")
        layer_cols=st.columns(6)
        show_candles=layer_cols[0].toggle("Candles",value=True,key="tech_show_candles")
        show_bb=layer_cols[1].toggle("Bollinger",value=True,key="tech_show_bb")
        show_vwap=layer_cols[2].toggle("VWAP",value=True,key="tech_show_vwap")
        show_ssl=layer_cols[3].toggle("SSL",value=True,key="tech_show_ssl")
        show_trend=layer_cols[4].toggle("HA Trend",value=True,key="tech_show_trend")
        show_adx=layer_cols[5].toggle("ADX / DI",value=True,key="tech_show_adx")
        pattern_col,band_col=st.columns([1,2])
        show_patterns=pattern_col.toggle("Candlestick patterns",value=False,key="tech_show_patterns")
        selected_bands=band_col.multiselect("Bollinger levels",[1,2,3],default=[1,2],key="tech_bb_layers")

        with st.expander("⚙ Indicator Parameters",expanded=False):
            bb_tab,vwap_tab,ssl_tab,trend_tab,adx_tab=st.tabs(["BB","VWAP","SSL Hybrid","HA Trend","ADX / DI"])
            with bb_tab:
                bb_col=st.columns(4)
                bb_length=bb_col[0].number_input("BB length",min_value=2,max_value=200,value=20,key="tech_bb_length")
                bb_multipliers=[bb_col[i].number_input(f"Level {i} standard deviations",min_value=.1,max_value=10.0,
                    value=float(i),step=.1,key=f"tech_bb_mult_{i}") for i in (1,2,3)]
            with vwap_tab:
                vw_col=st.columns(3)
                default_anchor=("Session" if technical_tf in ("1M","5M","15M","30M","1H","4H")
                    else "Month" if technical_tf=="1W" else "Year" if technical_tf=="1MO"
                    else "All" if technical_tf=="1Y" else "Week")
                vwap_anchors=["Session","Week","Month","Quarter","Year","All"]
                vwap_anchor=vw_col[0].selectbox("Anchor period",vwap_anchors,
                    index=vwap_anchors.index(default_anchor),key="tech_vwap_anchor")
                vwap_band=vw_col[1].number_input("VWAP σ multiplier",min_value=.1,max_value=5.0,value=1.0,step=.1,key="tech_vwap_band")
                show_vwap_band=vw_col[2].toggle("Show VWAP bands",value=False,key="tech_vwap_band_show")
            with ssl_tab:
                ssl_col=st.columns(4)
                ssl_ma_types=["HMA","JMA","EMA","SMA","WMA","DEMA","TEMA","VWMA"]
                ssl_base_type=ssl_col[0].selectbox("Baseline MA",ssl_ma_types,key="tech_ssl_base_type")
                ssl_base_length=ssl_col[1].number_input("Baseline length",min_value=2,max_value=200,value=60,key="tech_ssl_base_len")
                ssl2_type=ssl_col[2].selectbox("SSL2 MA",ssl_ma_types,index=1,key="tech_ssl2_type")
                ssl2_length=ssl_col[3].number_input("SSL2 length",min_value=2,max_value=100,value=5,key="tech_ssl2_len")
                ssl_col2=st.columns(4)
                ssl_exit_type=ssl_col2[0].selectbox("Exit MA",ssl_ma_types,key="tech_ssl_exit_type")
                ssl_exit_length=ssl_col2[1].number_input("Exit length",min_value=2,max_value=100,value=15,key="tech_ssl_exit_len")
                ssl_channel=ssl_col2[2].number_input("Channel multiplier",min_value=0.0,max_value=5.0,value=.2,step=.05,key="tech_ssl_channel")
                ssl_atr_length=ssl_col2[3].number_input("ATR length",min_value=2,max_value=100,value=14,key="tech_ssl_atr_len")
                jma_col=st.columns(2)
                ssl_jma_phase=jma_col[0].number_input("JMA Phase",min_value=-100,max_value=100,value=3,key="tech_jma_phase")
                ssl_jma_power=jma_col[1].number_input("JMA Power",min_value=1,max_value=10,value=1,key="tech_jma_power")
                ssl_criterion=jma_col[0].number_input("Continuation ATR factor",min_value=.1,max_value=3.0,value=.9,step=.1,key="tech_ssl_atr_criterion")
                show_ssl_signals=jma_col[1].toggle("Show SSL continuation / exit markers",value=False,key="tech_ssl_signals")
                ssl_layers=st.multiselect("SSL layers",["Baseline","Channel","SSL1","SSL2","Exit"],
                    default=["Baseline","Channel","SSL2"],key="tech_ssl_layers")
            with trend_tab:
                trend_col=st.columns(4)
                trend_type=trend_col[0].selectbox("HA smoothing MA",["EMA","HMA","ALMA","SMA","SWMA","WMA","VWMA","ZLEMA"],key="tech_trend_type")
                trend_length=trend_col[1].number_input("Trend length",min_value=1,max_value=200,value=9,key="tech_trend_len")
                alma_offset=trend_col[2].number_input("ALMA offset",min_value=0.0,max_value=1.0,value=.85,step=.05,key="tech_alma_offset")
                alma_sigma=trend_col[3].number_input("ALMA σ",min_value=1.0,max_value=20.0,value=6.0,step=.5,key="tech_alma_sigma")
                show_trend_range=st.toggle("Show HA high / low rails",value=False,key="tech_trend_range")
            with adx_tab:
                adx_col=st.columns(2)
                adx_length=adx_col[0].number_input("ADX / DI length",min_value=2,max_value=100,value=14,key="tech_adx_len")
                adx_threshold=adx_col[1].number_input("Trend threshold",min_value=1.0,max_value=80.0,value=22.0,step=1.0,key="tech_adx_threshold")

        try:
            candle_history=load_technical_ohlcv(single_ticker,technical_tf)
        except Exception as exc:
            candle_history=pd.DataFrame()
            st.warning(f"{single_ticker} {technical_tf} OHLCV download failed: {exc}")
        if candle_history.empty:
            st.info("No candles are available for this timeframe. Intraday views use only history permitted by the provider and never substitute daily bars.")
        else:
            candle_history=candle_history.dropna(subset=["Open","High","Low","Close"])
            chart_slice=candle_history if chart_bars=="All" else candle_history.tail(int(chart_bars))
            indicator_min=max(bb_length,ssl_base_length,ssl_exit_length,adx_length*2)
            if len(candle_history)<indicator_min:
                st.caption(f"Only {len(candle_history)} bars are available; approximately {indicator_min} are required to warm up the longest indicator. Unformed series remain blank.")
            bb_levels={level:bollinger_bands(candle_history["Close"],bb_length,bb_multipliers[level-1])
                       for level in selected_bands} if show_bb else {}
            vwap_values=anchored_vwap(candle_history,vwap_anchor) if show_vwap else pd.DataFrame()
            ssl_values=ssl_hybrid(candle_history,ssl_base_length,ssl_base_type,ssl2_length,ssl2_type,
                ssl_exit_length,ssl_exit_type,ssl_channel,ssl_atr_length,
                ssl_jma_phase,ssl_jma_power,ssl_criterion) if show_ssl else pd.DataFrame()
            trend_values=heikin_trend(candle_history,trend_length,trend_type,alma_offset,alma_sigma) if show_trend else pd.DataFrame()
            adx_values=adx_di(candle_history,adx_length) if show_adx else pd.DataFrame()
            pattern_values=candle_patterns(candle_history) if show_patterns else pd.DataFrame()
            rows=1+int(show_adx)+int(show_trend)
            heights=([.68,.2,.12] if rows==3 else [.78,.22] if rows==2 else [1.0])
            multi_fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=heights)
            if show_candles:
                multi_fig.add_trace(go.Candlestick(x=chart_slice.index,open=chart_slice["Open"],high=chart_slice["High"],
                    low=chart_slice["Low"],close=chart_slice["Close"],name=f"{single_ticker} Candles",
                    increasing_line_color="#00c3a5",decreasing_line_color="#ff4774"),row=1,col=1)
            else:
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=chart_slice["Close"],name=f"{single_ticker} Close",
                    line=dict(color="#38bdf8",width=2)),row=1,col=1)
            for level,bb_frame in bb_levels.items():
                color={1:"#38bdf8",2:"#a78bfa",3:"#64748b"}[level]
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=bb_frame["upper"].reindex(chart_slice.index),
                    name=f"BB{level} Upper",line=dict(color=color,width=1)),row=1,col=1)
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=bb_frame["lower"].reindex(chart_slice.index),
                    name=f"BB{level} Lower",line=dict(color=color,width=1),fill="tonexty",
                    fillcolor={1:"rgba(56,189,248,.07)",2:"rgba(167,139,250,.05)",3:"rgba(100,116,139,.04)"}[level]),row=1,col=1)
            if show_vwap:
                vwap_plot=vwap_values.reindex(chart_slice.index)
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=vwap_plot["vwap"],name=f"VWAP · {vwap_anchor}",
                    line=dict(color="#00e5ff",width=2)),row=1,col=1)
                if show_vwap_band:
                    for sign,label in ((1,"Upper"),(-1,"Lower")):
                        multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=vwap_plot["vwap"]+sign*vwap_band*vwap_plot["std"],
                            name=f"VWAP {label} σ",line=dict(color="#00a7b7",width=1,dash="dot")),row=1,col=1)
            if show_ssl:
                ssl_colors={"Baseline":"#f6c453","Channel":"#f6c453","SSL1":"#4ade80","SSL2":"#e879f9","Exit":"#ff8a60"}
                for label,column in (("Baseline","baseline"),("Channel","channel_upper"),("Channel","channel_lower"),
                                     ("SSL1","ssl1"),("SSL2","ssl2"),("Exit","exit")):
                    if label not in ssl_layers: continue
                    multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=ssl_values[column].reindex(chart_slice.index),
                        name=f"SSL {column}",line=dict(color=ssl_colors[label],width=2 if column=="baseline" else 1.2,
                            dash="dot" if label=="Channel" else "solid")),row=1,col=1)
                if show_ssl_signals:
                    ssl_marks=ssl_values.reindex(chart_slice.index).tail(100)
                    for column,label,color,symbol,side in (("buy_cont","SSL Buy Continuation","#00d7a3","triangle-up","Low"),
                        ("sell_cont","SSL Sell Continuation","#ff4774","triangle-down","High"),
                        ("exit_long","SSL Exit Long","#ffb452","x","High"),
                        ("exit_short","SSL Exit Short","#38bdf8","x","Low")):
                        marked=ssl_marks.index[ssl_marks[column].fillna(False)]
                        if len(marked):
                            multi_fig.add_trace(go.Scatter(x=marked,y=chart_slice.loc[marked,side],name=label,
                                mode="markers",marker=dict(color=color,symbol=symbol,size=10)),row=1,col=1)
            if show_trend:
                trend_plot=trend_values.reindex(chart_slice.index)
                bullish=trend_plot["score"]>=0
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=trend_plot["close"].where(bullish),name="HA Trend ↑",
                    line=dict(color="#26d6b0",width=2.4)),row=1,col=1)
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=trend_plot["close"].where(~bullish),name="HA Trend ↓",
                    line=dict(color="#ff5376",width=2.4)),row=1,col=1)
                if show_trend_range:
                    for column in ("high","low"):
                        multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=trend_plot[column],name=f"HA {column}",
                            line=dict(color="#64748b",width=1,dash="dot")),row=1,col=1)
            if show_patterns:
                recent_patterns=pattern_values.reindex(chart_slice.index).tail(80)
                for label,color,direction in (("Bullish Engulfing","#00e5a8",1),("Hammer","#f5c451",1),
                    ("Bearish Engulfing","#ff4774",-1),("Shooting Star","#ff9d4d",-1),("Doji","#b4a2ff",0)):
                    marked=recent_patterns.index[recent_patterns[label]]
                    if len(marked)==0: continue
                    anchor=chart_slice.loc[marked,"Low" if direction>=0 else "High"]
                    multi_fig.add_trace(go.Scatter(x=marked,y=anchor,mode="markers",name=label,
                        marker=dict(symbol="triangle-up" if direction>0 else "triangle-down" if direction<0 else "diamond",
                            color=color,size=11,line=dict(color="#0e1117",width=1))),row=1,col=1)
            row_number=2
            if show_adx:
                adx_plot=adx_values.reindex(chart_slice.index)
                for label,color in (("ADX","#f6c453"),("+DI","#00d7a3"),("-DI","#ff4774")):
                    multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=adx_plot[label],name=f"{label} {adx_length}",
                        line=dict(color=color,width=1.8 if label=="ADX" else 1.3)),row=row_number,col=1)
                multi_fig.add_hline(y=adx_threshold,line_dash="dot",line_color="#9aa6b7",row=row_number,col=1)
                multi_fig.update_yaxes(title="ADX / DI",row=row_number,col=1)
                row_number+=1
            if show_trend:
                multi_fig.add_trace(go.Scatter(x=chart_slice.index,y=trend_values["score"].reindex(chart_slice.index),
                    name="HA Trend Score",line=dict(color="#a78bfa",width=1.7)),row=row_number,col=1)
                multi_fig.add_hline(y=0,line_dash="dot",line_color="#9aa6b7",row=row_number,col=1)
                multi_fig.update_yaxes(title="HA Score",row=row_number,col=1)
            base_layout(multi_fig,790 if rows==3 else 680 if rows==2 else 540,"x unified")
            multi_fig.update_layout(title=dict(text=f"{single_ticker} · {technical_tf} · Multi-Indicator Workstation",
                font=dict(size=13,color="#f0a13a")),xaxis_rangeslider_visible=False,dragmode="pan",
                legend=dict(orientation="h",y=1.02,x=0,font=dict(size=10)))
            multi_fig.update_yaxes(title="Price",row=1,col=1,fixedrange=False)
            multi_fig.update_xaxes(fixedrange=False)
            st.plotly_chart(multi_fig,width="stretch",config={"displaylogo":False,"scrollZoom":True,
                "modeBarButtonsToAdd":["drawline","eraseshape"]})
            if show_patterns:
                findings=[f"{when:%Y-%m-%d %H:%M} · {label}" for when,row in pattern_values.tail(30).iterrows()
                          for label,found in row.items() if found]
                st.caption("Recent candlestick patterns: "+("; ".join(findings[-8:]) if findings else "none in the latest 30 bars"))
            st.caption(f"Coverage: {candle_history.index[0]:%Y-%m-%d %H:%M}—{candle_history.index[-1]:%Y-%m-%d %H:%M} ET · "
                f"{len(candle_history)} bars. Indicators are calculated on all available history before the chart window is sliced. 1M is usually limited to 7 days and 5M–30M to roughly 60 days. "
                "4H bars aggregate regular-session 1H bars by trading day; yearly bars aggregate daily history.")
    st.caption("Signals are research classifications based on relative strength, moving averages and statistical distance. They are not deterministic bottom forecasts or investment advice.")

with tab_pattern:
    regime_head,regime_control_1,regime_control_2,regime_control_3=st.columns([4.6,1.35,1.05,1.05],vertical_alignment="bottom")
    with regime_head:
        section_header("REGIME & PATTERN DETECTION","Trend Regime & Geometric Pattern Detection",
                       "Full history remains on the chart; 6M / 1Y / 3Y only set the initial viewport and can be dragged back at any time.")
    with regime_control_1:
        selected_pair=st.selectbox("Trading Pair",list(REGIME_PAIRS),key="regime_pair")
    base_symbol,leveraged_symbol=REGIME_PAIRS[selected_pair]
    with regime_control_2:
        target_symbol=st.selectbox("Analyze",[base_symbol,leveraged_symbol],index=1,key="regime_target")
    with regime_control_3:
        regime_horizon=st.selectbox("Chart Range",["6M","1Y","3Y","MAX"],index=3,key="regime_horizon")
    if not SCIPY_AVAILABLE:
        st.caption("⚠ SciPy is unavailable, so the safe fallback extrema routine is active. Installing requirements.txt enables scipy.signal.argrelextrema.")

    regime_data=pd.DataFrame()
    try:
        regime_data=load_ohlcv((base_symbol,leveraged_symbol),"max")
    except Exception as exc:
        st.warning(f"OHLCV download failed: {exc}")
    available_regime=set(regime_data.columns.get_level_values(0)) if not regime_data.empty else set()
    if regime_data.empty or any(t not in available_regime for t in (base_symbol,leveraged_symbol)):
        st.warning("The selected instruments did not return sufficient OHLCV history.")
    else:
        indicator_frame=regime_indicators(regime_data[target_symbol])
        regime_state,regime_stats=classify_regime(indicator_frame)
        patterns=detect_geometric_patterns(indicator_frame)
        if not regime_stats:
            st.warning("The regime engine requires at least 200 valid trading sessions.")
        else:
            detected=[]
            if patterns.get("trend_breakout"): detected.append("🚀 Descending Trendline Break")
            double_bottom=patterns.get("double_bottom")
            if double_bottom and double_bottom["breakout"]: detected.append("🔥 Double-Bottom Breakout")
            triangle=patterns.get("triangle")
            if triangle:
                triangle_signal=triangle["type"]
                if triangle["breakout_up"]: triangle_signal="🚀 "+triangle_signal+" Breakout"
                elif triangle["breakout_down"]: triangle_signal="⚠️ "+triangle_signal+" Breakdown"
                detected.append("🔺 "+triangle_signal)
            if patterns.get("channel"): detected.append("📉 Descending Channel")
            if not detected: detected.append("No confirmed breakout pattern")
            latest=float(regime_stats["close"]); ema50=float(regime_stats["ema50"])
            ema50_distance=(latest/ema50-1)*100
            pair_close=regime_data[leveraged_symbol]["Close"].div(regime_data[base_symbol]["Close"]).dropna()
            pair_momentum=last_return(pair_close,20)
            cards=st.columns(6)
            cards[0].metric("CURRENT REGIME",regime_state,delta_color="off")
            cards[1].metric("PATTERN SIGNAL"," · ".join(detected),delta_color="off")
            cards[2].metric("ADX 14",f"{regime_stats['adx']:.2f}",f"+DI {regime_stats['plus_di']:.1f} / −DI {regime_stats['minus_di']:.1f}",delta_color="off")
            cards[3].metric("VS EMA50",f"{ema50_distance:+.2f}%")
            cards[4].metric("BBW 20",f"{regime_stats['bbw']*100:.2f}%",delta_color="off")
            cards[5].metric(f"{leveraged_symbol}/{base_symbol} · 1M",format_num(pair_momentum),delta_color="normal")

            # Keep every downloaded bar in the traces. The horizon only controls the initial viewport.
            plotted=indicator_frame.copy()
            initial_range=chart_initial_range(plotted.index,regime_horizon)
            pattern_fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.06,row_heights=[.78,.22])
            pattern_fig.add_trace(go.Candlestick(x=plotted.index,open=plotted["Open"],high=plotted["High"],low=plotted["Low"],close=plotted["Close"],
                name=target_symbol,increasing_line_color="#24c990",decreasing_line_color="#ee5266",increasing_fillcolor="#168b67",decreasing_fillcolor="#a72c42"),row=1,col=1)
            for name,column,color,width in [("EMA20","EMA20","#35d5ff",1.5),("EMA50","EMA50","#f0a13a",1.7),("SMA200","SMA200","#a8adb5",1.5)]:
                pattern_fig.add_trace(go.Scatter(x=plotted.index,y=plotted[column],name=name,line=dict(color=color,width=width)),row=1,col=1)
            if regime_state.startswith("📦"):
                box_start=indicator_frame.index[-30]
                pattern_fig.add_shape(type="rect",x0=box_start,x1=indicator_frame.index[-1],y0=regime_stats["box_low"],y1=regime_stats["box_high"],
                    line=dict(color="#e8bd55",width=1.5,dash="dash"),fillcolor="rgba(232,189,85,.08)",row=1,col=1)
                pattern_fig.add_annotation(x=indicator_frame.index[-1],y=regime_stats["box_high"],text="30D RESISTANCE",showarrow=False,xanchor="right",font=dict(color="#e8bd55",size=10),row=1,col=1)
            recent=patterns["recent"]
            trend=patterns.get("trend")
            if trend:
                trend_dates=recent.index[trend["x"]]
                pattern_fig.add_trace(go.Scatter(x=trend_dates,y=trend["y"],name="Descending Trendline",
                    line=dict(color="#31d6a0" if patterns["trend_breakout"] else "#e8bd55",width=2.4,dash="dash")),row=1,col=1)
                high_idx=trend["idx"]
                pattern_fig.add_trace(go.Scatter(x=recent.index[high_idx],y=recent["High"].iloc[high_idx],name="Lower Highs",mode="markers",
                    marker=dict(color="#e8bd55",size=9,symbol="diamond")),row=1,col=1)
                if patterns["trend_breakout"]:
                    pattern_fig.add_annotation(x=recent.index[-1],y=recent["High"].iloc[-1],text="🚀 TRENDLINE BREAK",showarrow=True,
                        arrowhead=2,arrowcolor="#31d6a0",ax=-55,ay=48,font=dict(color="#31d6a0",size=11),row=1,col=1)
            channel=patterns.get("channel")
            if channel:
                channel_dates=recent.index[channel["x"]]
                pattern_fig.add_trace(go.Scatter(x=channel_dates,y=channel["upper"],name="Channel Upper",line=dict(color="#38c6d9",width=1.5)),row=1,col=1)
                pattern_fig.add_trace(go.Scatter(x=channel_dates,y=channel["lower"],name="Channel Lower",line=dict(color="#38c6d9",width=1.5),fill="tonexty",fillcolor="rgba(56,198,217,.07)"),row=1,col=1)
            if double_bottom:
                w_idx=double_bottom["idx"]
                w_dates=[recent.index[i] for i in w_idx]
                w_prices=list(double_bottom["prices"])
                if double_bottom["breakout"]: w_dates.append(recent.index[-1]); w_prices.append(float(recent["Close"].iloc[-1]))
                pattern_fig.add_trace(go.Scatter(x=w_dates,y=w_prices,name="W Bottom",mode="lines+markers",
                    line=dict(color="#27d3e2",width=3),marker=dict(size=9,color="#27d3e2")),row=1,col=1)
                pattern_fig.add_hline(y=double_bottom["neckline"],line_dash="dash",line_color="#d18cff",line_width=2,
                    annotation_text="W NECKLINE",annotation_font_color="#d18cff",row=1,col=1)
            if triangle:
                triangle_dates=recent.index[triangle["x"]]
                triangle_color="#31d6a0" if triangle["breakout_up"] else "#ff5874" if triangle["breakout_down"] else "#f6c453"
                pattern_fig.add_trace(go.Scatter(x=triangle_dates,y=triangle["upper"],name=f"{triangle['type']} · Upper",
                    line=dict(color=triangle_color,width=2.6)),row=1,col=1)
                pattern_fig.add_trace(go.Scatter(x=triangle_dates,y=triangle["lower"],name=f"{triangle['type']} · Lower",
                    line=dict(color=triangle_color,width=2.6),fill="tonexty",fillcolor="rgba(246,196,83,.075)"),row=1,col=1)
                pattern_fig.add_trace(go.Scatter(x=recent.index[triangle["high_idx"]],y=recent["High"].iloc[triangle["high_idx"]],
                    name="Triangle Highs",mode="markers",marker=dict(color=triangle_color,size=8,symbol="triangle-down")),row=1,col=1)
                pattern_fig.add_trace(go.Scatter(x=recent.index[triangle["low_idx"]],y=recent["Low"].iloc[triangle["low_idx"]],
                    name="Triangle Lows",mode="markers",marker=dict(color=triangle_color,size=8,symbol="triangle-up")),row=1,col=1)
                if triangle["breakout_up"] or triangle["breakout_down"]:
                    arrow_text="🚀 TRIANGLE BREAKOUT" if triangle["breakout_up"] else "⚠ TRIANGLE BREAKDOWN"
                    arrow_y=float(recent["High"].iloc[-1] if triangle["breakout_up"] else recent["Low"].iloc[-1])
                    pattern_fig.add_annotation(x=recent.index[-1],y=arrow_y,text=arrow_text,showarrow=True,arrowhead=2,
                        arrowcolor=triangle_color,ax=-65,ay=45 if triangle["breakout_up"] else -45,
                        font=dict(color=triangle_color,size=11),row=1,col=1)
            pattern_fig.add_trace(go.Scatter(x=plotted.index,y=plotted["ADX14"],name="ADX14",line=dict(color="#f0a13a",width=2)),row=2,col=1)
            pattern_fig.add_trace(go.Scatter(x=plotted.index,y=plotted["+DI"],name="+DI",line=dict(color="#31d6a0",width=1.3)),row=2,col=1)
            pattern_fig.add_trace(go.Scatter(x=plotted.index,y=plotted["-DI"],name="−DI",line=dict(color="#ff5874",width=1.3)),row=2,col=1)
            pattern_fig.add_hline(y=22,line_dash="dot",line_color="#8593a5",annotation_text="ADX 22",row=2,col=1)
            base_layout(pattern_fig,720,"x unified")
            pattern_fig.update_layout(title=dict(text=f"{target_symbol} · Regime & Geometric Pattern Engine",font=dict(size=13,color="#f0a13a")),
                                      dragmode="zoom",uirevision=f"regime-{target_symbol}",xaxis_rangeslider_visible=False)
            if initial_range:
                pattern_fig.update_xaxes(range=initial_range,autorange=False)
            else:
                pattern_fig.update_xaxes(autorange=True)
            pattern_fig.update_xaxes(fixedrange=False,showspikes=True,spikemode="across",spikesnap="cursor")
            pattern_fig.update_yaxes(fixedrange=False,showspikes=True,spikesnap="cursor")
            pattern_fig.update_xaxes(rangeselector=dict(
                buttons=[dict(count=6,label="6M",step="month",stepmode="backward"),
                         dict(count=1,label="1Y",step="year",stepmode="backward"),
                         dict(count=3,label="3Y",step="year",stepmode="backward"),
                         dict(label="MAX",step="all")],
                bgcolor="#0b1421",activecolor="#273b55",bordercolor="#26384e",borderwidth=1,
                font=dict(color="#9fb0c5",size=10),x=.72,xanchor="left",y=1.13,yanchor="top"),row=1,col=1)
            pattern_fig.update_yaxes(title="Price",row=1,col=1); pattern_fig.update_yaxes(title="ADX / DI",row=2,col=1)
            st.plotly_chart(pattern_fig,width="stretch",config={"displaylogo":False,"scrollZoom":True,
                "doubleClick":"reset","modeBarButtonsToAdd":["drawline","eraseshape"]})

            # Geometry-derived research levels; no long-entry prompt is issued in an active downtrend break.
            recent30=indicator_frame.tail(30); entry=float(regime_stats["ema20"]); stop=float(recent30["Low"].min())
            rationale="EMA20 retest; invalid below the 30-day low"
            geometry=max(entry-stop,float(regime_stats["atr"]))
            if double_bottom and double_bottom["breakout"]:
                entry=float(double_bottom["neckline"]); stop=float(double_bottom["low2"]); geometry=entry-stop
                rationale="Neckline retest; Low2 is the structural defense"
            elif patterns.get("trend_breakout") and trend:
                entry=float(trend["level"]); stop=float(recent30["Low"].min())
                geometry=float(channel["height"]) if channel and float(channel["height"])>0 else max(entry-stop,float(regime_stats["atr"]))
                rationale="Retest after a descending-trendline break"
            elif triangle and triangle["breakout_up"]:
                entry=float(triangle["upper"][-1]); stop=float(triangle["lower"][-1])
                geometry=max(float(triangle["height"]),entry-stop,float(regime_stats["atr"]))
                rationale=f"{triangle['type']} upper-bound retest; lower bound is structural defense"
            elif regime_state.startswith("📦"):
                box_range=float(regime_stats["box_high"]-regime_stats["box_low"])
                entry=float(regime_stats["box_low"]+.15*box_range); stop=float(regime_stats["box_low"]-.25*regime_stats["atr"]); geometry=box_range
                rationale="Range-entry zone 15% above the lower boundary"
            target1=float(entry+geometry/3)
            level_cols=st.columns(3)
            level_cols[0].metric("TACTICAL ENTRY",f"${entry:,.2f}",rationale,delta_color="off")
            level_cols[1].metric("HARD DEFENSE",f"${stop:,.2f}","A break invalidates the geometry",delta_color="inverse")
            level_cols[2].metric("FIRST TARGET · 1/3",f"${target1:,.2f}","One third of the detected structure height",delta_color="off")
            if regime_state.startswith("⚠️"):
                st.error("Active downtrend break: the tactical level is observation only. Reassess after price recovers EMA50 and ADX/DI improves.")
            elif double_bottom and double_bottom["breakout"]:
                st.success(f"🔥 Double bottom is above the ${double_bottom['neckline']:.2f} neckline; Low2 at ${double_bottom['low2']:.2f} is the structural defense.")
            elif patterns.get("trend_breakout"):
                st.success("🚀 The latest close crossed the descending trendline. Monitor whether the breakout level holds on a retest.")
            elif triangle and (triangle["breakout_up"] or triangle["breakout_down"]):
                direction="upside breakout" if triangle["breakout_up"] else "downside breakdown"
                if triangle["breakout_up"]:
                    st.success(f"{triangle['type']} {direction}; the converging boundaries and breakout bar are highlighted.")
                else:
                    st.error(f"{triangle['type']} {direction}; wait for a lower-bound recovery or further risk release.")
            else:
                st.info("No new geometric breakout is confirmed. Regime cards and levels are observation tools, not standalone trade instructions.")
            st.caption("Chart interaction: box-select to zoom both axes; drag an axis to scale it independently; use the wheel to zoom and double-click to restore full history.")
            st.caption("Automated pattern recognition can lag or misclassify. Leveraged ETFs are affected by daily rebalancing, volatility decay and path dependence.")

if False:  # Option UOA / Gamma module intentionally excluded from this edition.
    option_head,option_control,moneyness_control=st.columns([4.4,1.1,1.7],vertical_alignment="bottom")
    with option_head:
        section_header("OPTIONS INTELLIGENCE","Options UOA & Gamma · 期权异动雷达","扫描7–30 DTE 虚值合约；所有链路均独立容错并受15分钟缓存保护。")
    with option_control:
        option_symbol=st.selectbox("Underlying",OPTION_POOL,key="option_radar_symbol",
                                   format_func=lambda symbol:OPTION_LABELS.get(symbol,symbol))
    with moneyness_control:
        moneyness_pct=st.slider("行权价偏离范围 · Strike Moneyness",10,30,15,5,format="±%d%%",key="uoa_moneyness")
    try:
        chain,spot=load_option_snapshot(option_symbol,4,allow_saved_snapshot=True)
    except Exception:
        chain,spot=pd.DataFrame(),np.nan
    unusual=unusual_option_rows(chain,spot,1.5,1000,moneyness_pct/100)
    skew_ratio,skew_curve=option_skew_proxy(chain,spot)
    option_cards=st.columns(4)
    option_cards[0].metric("SPOT","N/A" if not np.isfinite(spot) else f"${spot:,.2f}")
    option_cards[1].metric("UOA CONTRACTS",f"{len(unusual)}",">=1.5× OI · Vol>=1000",delta_color="off")
    option_cards[2].metric("±4% OTM IV SKEW","N/A" if not np.isfinite(skew_ratio) else f"{skew_ratio:.3f}",
                           "⚡ 倒挂警戒" if np.isfinite(skew_ratio) and skew_ratio>=.95 else "Call IV / Put IV",delta_color="off")
    chain_source=chain.attrs.get("source","UNAVAILABLE") if not chain.empty else "UNAVAILABLE"
    option_cards[3].metric("CHAIN STATUS",chain_source,f"{chain['Expiration'].nunique() if not chain.empty else 0} expiries",delta_color="off")
    if chain.empty:
        st.warning(f"{option_symbol} 近月期权链暂不可用；Yahoo 限流或非交易时段可能导致空响应，请稍后重试。")
    else:
        asof_utc=chain.attrs.get("asof_utc","")
        st.caption(f"期权链来源：{chain_source} · 采集时间 UTC：{asof_utc}。历史快照仅供回看，不代表实时盘口或今日异动。")
        uoa_col,skew_col=st.columns(2,gap="medium")
        with uoa_col:
            section_header("VOL > OI SCANNER","异常大单扫描",f"仅保留现价 ±{moneyness_pct}% 内的虚值合约；Volume >= 1.5×OI 且 Volume >= 1000。")
            if unusual.empty:
                st.info("当前链未发现满足阈值的虚值合约。")
            else:
                unusual_style=unusual.head(60).style.format({"Strike":"${:,.2f}","Moneyness":"{:+.1f}%","Vol":"{:,.0f}","OI":"{:,.0f}","Vol/OI":"{:.2f}×","IV":"{:.1f}%"},na_rep="—")
                st.dataframe(unusual_style,width="stretch",hide_index=True,height=430)
        with skew_col:
            section_header("IV SKEW CURVE","Strike vs Implied Volatility","最近到期日；绿色为 Call IV，橙红为 Put IV。")
            skew_fig=go.Figure()
            for option_type,color in (("Call","#31d6a0"),("Put","#ff7857")):
                curve=skew_curve[skew_curve["Type"].eq(option_type)].sort_values("strike") if not skew_curve.empty else pd.DataFrame()
                if curve.empty: continue
                skew_fig.add_trace(go.Scatter(x=curve["strike"],y=curve["impliedVolatility"]*100,name=f"{option_type} IV",
                    mode="lines+markers",line=dict(color=color,width=2),marker=dict(size=5),hovertemplate="$%{x:.2f}<br>IV %{y:.1f}%<extra></extra>"))
            base_layout(skew_fig,430,"x unified"); skew_fig.add_vline(x=spot,line_dash="dash",line_color="#f4f7fb",annotation_text=f"SPOT {spot:.2f}")
            skew_fig.update_xaxes(title="Strike"); skew_fig.update_yaxes(title="Implied Volatility",ticksuffix="%")
            st.plotly_chart(skew_fig,width="stretch",config={"displaylogo":False})

        section_header("STRIKE OI PROFILE","Gamma 引力位 · Open Interest by Strike","OI 仅作为 Gamma 磁吸代理；未包含做市商净头寸方向与合约 Gamma。")
        profile=chain[chain["strike"].between(spot*.70,spot*1.30)].copy()
        profile=profile.groupby(["strike","Type"],as_index=False)["openInterest"].sum()
        oi_fig=go.Figure()
        for option_type,color,sign in (("Call","#31d6a0",1),("Put","#ff5874",-1)):
            part=profile[profile["Type"].eq(option_type)]
            oi_fig.add_trace(go.Bar(x=part["strike"],y=part["openInterest"]*sign,name=f"{option_type} OI",marker_color=color,
                hovertemplate=f"{option_type} $%{{x:.2f}}<br>OI %{{customdata:,.0f}}<extra></extra>",customdata=part["openInterest"]))
        base_layout(oi_fig,430); oi_fig.update_layout(barmode="relative")
        oi_fig.add_vline(x=spot,line_dash="dash",line_color="#ffffff",line_width=2,annotation_text=f"SPOT ${spot:.2f}")
        oi_fig.update_xaxes(title="Strike"); oi_fig.update_yaxes(title="Call OI (+) / Put OI (−)")
        st.plotly_chart(oi_fig,width="stretch",config={"displaylogo":False})
        section_header("FULL OPTION CHAIN","完整近月 Call / Put 期权链","选择到期日和方向查看全部可用合约；快照数据明确标注采集时间。")
        chain_controls=st.columns([1,1,3])
        with chain_controls[0]:
            chosen_expiry=st.selectbox("到期日",sorted(chain["Expiration"].astype(str).unique()),key="full_chain_expiry")
        with chain_controls[1]:
            chosen_type=st.selectbox("方向",["Call","Put","All"],key="full_chain_type")
        shown=chain[chain["Expiration"].astype(str).eq(chosen_expiry)].copy()
        if chosen_type!="All": shown=shown[shown["Type"].eq(chosen_type)]
        shown=shown[shown["strike"].between(spot*.70,spot*1.30)] if np.isfinite(spot) else shown
        columns={"contractSymbol":"Contract","Type":"Type","strike":"Strike","bid":"Bid","ask":"Ask","lastPrice":"Last","volume":"Vol","openInterest":"OI","impliedVolatility":"IV"}
        for column in columns:
            if column not in shown: shown[column]=np.nan
        shown=shown[list(columns)].rename(columns=columns).sort_values("Strike")
        shown["IV"]=shown["IV"]*100
        st.dataframe(shown.style.format({"Strike":"${:,.2f}","Bid":"${:,.2f}","Ask":"${:,.2f}","Last":"${:,.2f}","Vol":"{:,.0f}","OI":"{:,.0f}","IV":"{:.1f}%"},na_rep="—"),width="stretch",hide_index=True,height=470)
        st.caption("方法限制：Yahoo 链不提供 Delta 与逐笔买卖方向，因此本页使用 ±4% OTM IV 作为 25-Delta Skew 近似；Volume/OI 不能证明主动买入或新开仓。")

with tab_flow:
    flow_head,flow_control=st.columns([5,1.2],vertical_alignment="bottom")
    with flow_head:
        section_header("PRIMARY MARKET FLOW","ETF Primary-Market Share Tracking","TQQQ / SOXL, QQQ / SPY and SOXX / SMH. Daily Flow = ΔShares × Prior NAV only when consecutive issuer observations exist.")
    with flow_control:
        flow_symbol=st.selectbox("ETF selector",list(PRIMARY_FLOW_ETFS),key="leveraged_flow_symbol",
            format_func=lambda value:PRIMARY_FLOW_ETFS[value]["label"])
    flow_frame=flow_snapshots.get(flow_symbol,pd.DataFrame())
    if flow_symbol not in flow_snapshots:
        try: flow_frame=load_primary_flow(flow_symbol)
        except Exception: flow_frame=pd.DataFrame()
    if flow_frame.empty:
        st.warning(f"{flow_symbol} price history is temporarily unavailable. The system will not estimate creations or redemptions from volume.")
    else:
        flow_meta=PRIMARY_FLOW_ETFS[flow_symbol]; extreme_threshold=float(flow_meta["threshold"])
        valid_flow=flow_frame.dropna(subset=["DailyFlow"]); latest_flow=float(valid_flow["DailyFlow"].iloc[-1])
        flow_source=str(flow_frame["Source"].iloc[-1]) if "Source" in flow_frame else "UNKNOWN"
        verified_shares=bool(flow_frame["HasVerifiedShares"].iloc[-1]) if "HasVerifiedShares" in flow_frame else False
        official_asof=pd.to_datetime(flow_frame["OfficialAsOf"].iloc[-1],errors="coerce") if "OfficialAsOf" in flow_frame else pd.NaT
        logger_status=str(flow_frame["LoggerStatus"].iloc[-1]) if "LoggerStatus" in flow_frame else "LOGGER UNKNOWN"
        record_count=int(flow_frame["RecordCount"].iloc[-1]) if "RecordCount" in flow_frame else 0
        window=valid_flow.tail(60); close_low=float(window["Close"].min()); close_high=float(window["Close"].max())
        price_low_zone=float(window["Close"].iloc[-1])<=close_low+.20*max(close_high-close_low,0)
        latest_is_official=bool(window["HasVerifiedFlow"].iloc[-1]) if "HasVerifiedFlow" in window else False
        flow_new_high=latest_flow>0 and latest_flow>=float(window["DailyFlow"].max())
        divergence=verified_shares and latest_is_official and price_low_zone and flow_new_high
        def compact_dollar(value: float) -> str:
            if not np.isfinite(value): return "N/A"
            return f"${value/1e9:+,.2f}B" if abs(value)>=1e9 else f"${value/1e6:+,.1f}M"
        flow_cards=st.columns(4)
        has_verified_flow=bool(window["HasVerifiedFlow"].iloc[-1]) if "HasVerifiedFlow" in window else False
        any_verified_flow=bool(window["HasVerifiedFlow"].any()) if "HasVerifiedFlow" in window else False
        flow_cards[0].metric("LATEST VERIFIED FLOW",compact_dollar(latest_flow) if has_verified_flow else "AWAITING CONSECUTIVE DISCLOSURE")
        flow_cards[1].metric("60D MAX INFLOW",compact_dollar(float(window["DailyFlow"].max())) if any_verified_flow else "AWAITING CONSECUTIVE DISCLOSURE")
        flow_cards[2].metric("PRICE · 60D POSITION",f"${window['Close'].iloc[-1]:,.2f}","LOW ZONE" if price_low_zone else "MID / HIGH RANGE",delta_color="off")
        flow_cards[3].metric("DIVERGENCE","🔥 EXTREME BOTTOM DIVERGENCE" if divergence else "NO CONFIRMATION",delta_color="off")
        if verified_shares:
            asof_text=official_asof.strftime("%Y-%m-%d") if pd.notna(official_asof) else "N/A"
            st.success(f"Data layer: {flow_source} · shares/NAV through {asof_text} · {record_count:,} valid records · {logger_status}. Unpublished dates remain placeholders.")
            if not any_verified_flow:
                st.warning("No consecutive issuer-share observations yet. Zero-height bars are placeholders, not confirmed zero net flow; price and published NAV remain visible.")
        else:
            st.warning(f"{flow_symbol} has no verifiable daily issuer shares/NAV history. Placeholder bars cannot be interpreted as primary-market flow.")
        flow_fig=make_subplots(specs=[[{"secondary_y":True}]])
        scale=1e9 if max(float(flow_frame["DailyFlow"].abs().max()),extreme_threshold)>=1e9 else 1e6
        unit="B" if scale==1e9 else "M"; flow_scaled=flow_frame["DailyFlow"]/scale; threshold_scaled=extreme_threshold/scale
        bar_colors=np.where(flow_frame["DailyFlow"]>=extreme_threshold,"#f6c453",
            np.where(flow_frame["DailyFlow"]<=-extreme_threshold,"#ff365f",
                     np.where(flow_frame["DailyFlow"]>=0,"#31d6a0","#b9435a")))
        flow_status=np.where(flow_frame["HasVerifiedFlow"],"Verified consecutive issuer shares","Missing shares · placeholder zero")
        flow_fig.add_trace(go.Bar(x=flow_frame.index,y=flow_scaled,name="Primary Flow (unverified dates = 0)",
            marker_color=bar_colors,customdata=flow_status,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>Flow $%{{y:+,.2f}}{unit}<br>%{{customdata}}<extra></extra>"),secondary_y=False)
        flow_fig.add_trace(go.Scatter(x=flow_frame.index,y=flow_frame["Close"],name=f"{flow_symbol} Close",
            line=dict(color="#5aa2ff",width=2.2),hovertemplate="%{x|%Y-%m-%d}<br>Close $%{y:.2f}<extra></extra>"),secondary_y=True)
        if "NAV" in flow_frame and flow_frame["NAV"].notna().any():
            flow_fig.add_trace(go.Scatter(x=flow_frame.index,y=flow_frame["NAV"],name="Official NAV",
                line=dict(color="#f6c453",width=1.5,dash="dot"),hovertemplate="%{x|%Y-%m-%d}<br>NAV $%{y:.4f}<extra></extra>"),secondary_y=True)
        base_layout(flow_fig,600,"x unified")
        threshold_label=compact_dollar(extreme_threshold).replace("+","")
        flow_fig.add_hline(y=threshold_scaled,line_dash="dash",line_color="#f6c453",annotation_text=f"+{threshold_label} EXTREME")
        flow_fig.add_hline(y=-threshold_scaled,line_dash="dash",line_color="#ff5874",annotation_text=f"-{threshold_label} EXTREME")
        positive_events=flow_frame.loc[flow_frame["DailyFlow"].ge(extreme_threshold),"DailyFlow"]/scale
        negative_events=flow_frame.loc[flow_frame["DailyFlow"].le(-extreme_threshold),"DailyFlow"]/scale
        positive_text="🔥 STRATEGIC CORE INFLOW" if flow_symbol in ("QQQ","SPY") else "🔥 EXTREME CREATION"
        negative_text="⚠️ INSTITUTIONAL REDEMPTION" if flow_symbol in ("QQQ","SPY") else "⚠️ EXTREME REDEMPTION"
        for event_date,event_value in positive_events.tail(5).items():
            flow_fig.add_annotation(x=event_date,y=event_value,text=positive_text,
                showarrow=True,arrowhead=2,arrowcolor="#f6c453",font=dict(color="#f6c453",size=10),ax=0,ay=-32)
        for event_date,event_value in negative_events.tail(5).items():
            flow_fig.add_annotation(x=event_date,y=event_value,text=negative_text,
                showarrow=True,arrowhead=2,arrowcolor="#ff5874",font=dict(color="#ff7187",size=10),ax=0,ay=32)
        flow_fig.update_yaxes(title=f"Verified Creation / Redemption · ${unit}",secondary_y=False)
        flow_fig.update_yaxes(title="NAV / Adjusted Close",secondary_y=True)
        st.plotly_chart(flow_fig,width="stretch",config={"displaylogo":False,"scrollZoom":True})
        st.caption(f"Strict methodology: only consecutive shares outstanding and prior-day NAV are used. Missing or failed dates display placeholder zero and cannot be interpreted as confirmed zero flow. {flow_meta['class']} extreme threshold: {threshold_label}. Volume, return and signed-dollar-volume are excluded.")

with tab_circuit:
    section_header("SYSTEMIC RISK LOCKS","Circuit Breakers","Three independent safety locks: index 200-day trends, credit risk appetite and the VIX term structure.")
    risk_cards=st.columns(3)
    risk_cards[0].metric("200 SMA TREND LOCK","🚨 LOCKED" if breakers["broken"] else "🟢 OPEN",
                         "Broken: "+(", ".join(breakers["broken"]) if breakers["broken"] else "none"),delta_color="off")
    risk_cards[1].metric("CREDIT LIQUIDITY","⚠️ TIGHTENING" if breakers["credit_warning"] else "🟢 NORMAL",
                         format_num(float(breakers["credit_change"]))+" · HYG/LQD 20D",delta_color="off")
    risk_cards[2].metric("VIX TERM STRUCTURE","🚨 INVERTED" if breakers["vix_inverted"] else "🟢 CONTANGO / NORMAL",
                         "N/A" if not np.isfinite(float(breakers["vix_ratio"])) else f"VIX/VIX3M {breakers['vix_ratio']:.3f}",delta_color="off")
    if breakers["credit_warning"]: st.warning("⚠️ Credit liquidity is tightening: HYG/LQD is below its 50-day average with negative 20-day momentum.")
    if breakers["vix_inverted"]: st.error("🚨 Volatility term structure is inverted: VIX / VIX3M >= 1.0. Broad liquidation risk is elevated.")
    risk_left,risk_right=st.columns(2,gap="medium")
    with risk_left:
        section_header("TREND & CREDIT","Index Trend Lock / Credit Ratio","Normalized index trends and the HYG/LQD credit-risk appetite ratio.")
        risk_fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.10)
        for ticker,color in (("SPY","#31d6a0"),("QQQ","#4f86d9")):
            s=prices[ticker].dropna().tail(504); rebased=s/s.iloc[0]*100; sma=s.rolling(200,min_periods=200).mean()/s.iloc[0]*100
            risk_fig.add_trace(go.Scatter(x=s.index,y=rebased,name=ticker,line=dict(color=color,width=2)),row=1,col=1)
            risk_fig.add_trace(go.Scatter(x=s.index,y=sma,name=f"{ticker} 200SMA",line=dict(color=color,width=1,dash="dash")),row=1,col=1)
        credit=prices[["HYG","LQD"]].dropna(); credit_ratio=credit["HYG"].div(credit["LQD"]).tail(504)
        risk_fig.add_trace(go.Scatter(x=credit_ratio.index,y=credit_ratio,name="HYG/LQD",line=dict(color="#f0a13a",width=2)),row=2,col=1)
        risk_fig.add_trace(go.Scatter(x=credit_ratio.index,y=credit_ratio.rolling(50).mean(),name="50D",line=dict(color="#a8adb5",width=1.3,dash="dash")),row=2,col=1)
        base_layout(risk_fig,600,"x unified"); risk_fig.update_yaxes(title="Rebased 100",row=1,col=1); risk_fig.update_yaxes(title="HYG/LQD",row=2,col=1)
        st.plotly_chart(risk_fig,width="stretch",config={"displaylogo":False})
    with risk_right:
        section_header("VOLATILITY CURVE","VIX / VIX3M Term Structure","A ratio at or above 1 means spot volatility exceeds three-month volatility.")
        vix_pair=prices[["^VIX","^VIX3M"]].dropna().tail(504) if {"^VIX","^VIX3M"}.issubset(prices.columns) else pd.DataFrame()
        if vix_pair.empty:
            st.warning("VIX3M data is currently unavailable.")
        else:
            vol_fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.10)
            vol_fig.add_trace(go.Scatter(x=vix_pair.index,y=vix_pair["^VIX"],name="VIX",line=dict(color="#ff5874",width=2)),row=1,col=1)
            vol_fig.add_trace(go.Scatter(x=vix_pair.index,y=vix_pair["^VIX3M"],name="VIX3M",line=dict(color="#4f86d9",width=2)),row=1,col=1)
            term_ratio=vix_pair["^VIX"].div(vix_pair["^VIX3M"])
            vol_fig.add_trace(go.Scatter(x=term_ratio.index,y=term_ratio,name="VIX/VIX3M",fill="tozeroy",line=dict(color="#f0a13a",width=2)),row=2,col=1)
            vol_fig.add_hline(y=1,line_dash="dash",line_color="#ff5874",annotation_text="INVERSION",row=2,col=1)
            base_layout(vol_fig,600,"x unified"); vol_fig.update_yaxes(title="Vol Index",row=1,col=1); vol_fig.update_yaxes(title="Ratio",row=2,col=1,range=[max(.5,float(term_ratio.min())*.95),max(1.15,float(term_ratio.max())*1.05)])
            st.plotly_chart(vol_fig,width="stretch",config={"displaylogo":False})

with tab_fixed:
    section_header("RATES TERMINAL", "U.S. Treasury Yield Curve",
                   "13-week, 5-year, 10-year and 30-year yield indices available through Yahoo Finance; changes are shown in basis points.")
    yield_rows = []
    curve_snapshots = {"LATEST": [], "1W AGO": [], "1M AGO": [], "YTD START": []}
    maturities = list(YIELD_TICKERS)
    tenor_years = {"3M": .25, "5Y": 5, "10Y": 10, "30Y": 30}
    for maturity, ticker in YIELD_TICKERS.items():
        s = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
        if s.empty:
            continue
        year_data = s[s.index.year == s.index[-1].year]
        latest = float(s.iloc[-1])
        week = float(s.iloc[-6]) if len(s) > 5 else np.nan
        month = float(s.iloc[-22]) if len(s) > 21 else np.nan
        ytd = float(year_data.iloc[0]) if not year_data.empty else np.nan
        curve_snapshots["LATEST"].append((tenor_years[maturity], maturity, latest))
        curve_snapshots["1W AGO"].append((tenor_years[maturity], maturity, week))
        curve_snapshots["1M AGO"].append((tenor_years[maturity], maturity, month))
        curve_snapshots["YTD START"].append((tenor_years[maturity], maturity, ytd))
        yield_rows.append({"Maturity":maturity,"Ticker":ticker,"Yield":latest,
                           "1D (bps)":value_change_bps(s,1),"1W (bps)":value_change_bps(s,5),
                           "1M (bps)":value_change_bps(s,21),
                           "YTD (bps)":float((latest-ytd)*100) if np.isfinite(ytd) else np.nan})

    yield_df = pd.DataFrame(yield_rows)
    rate_cards = st.columns(4)
    for col, maturity in zip(rate_cards, maturities):
        row = yield_df[yield_df["Maturity"] == maturity]
        if row.empty:
            col.metric(maturity, "N/A", delta_color="off")
        else:
            col.metric(f"UST {maturity}", f"{row.iloc[0]['Yield']:.3f}%",
                       f"{row.iloc[0]['1D (bps)']:+.1f} bps", help=YIELD_TICKERS[maturity])

    curve_col, history_col = st.columns(2, gap="medium")
    with curve_col:
        section_header("TERM STRUCTURE", "Yield Curve Snapshots", "Latest, one week ago, one month ago and start-of-year curve snapshots.")
        curve_fig = go.Figure()
        curve_colors = {"LATEST":"#f0a13a","1W AGO":"#287bc1","1M AGO":"#a4a9b0","YTD START":"#27914c"}
        for label, points in curve_snapshots.items():
            valid = sorted([p for p in points if np.isfinite(p[2])])
            if not valid:
                continue
            curve_fig.add_trace(go.Scatter(x=[p[0] for p in valid],y=[p[2] for p in valid],mode="lines+markers",
                name=label,line=dict(color=curve_colors[label],width=2),marker=dict(size=6),
                customdata=[p[1] for p in valid],hovertemplate=f"<b>{label}</b><br>%{{customdata}} · %{{y:.3f}}%<extra></extra>"))
        base_layout(curve_fig,420,"x unified")
        curve_fig.update_xaxes(title="Maturity",tickvals=[.25,5,10,30],ticktext=["3M","5Y","10Y","30Y"],type="log")
        curve_fig.update_yaxes(title="Yield %",ticksuffix="%")
        st.plotly_chart(curve_fig,width="stretch",config={"displaylogo":False})
    with history_col:
        section_header("RATES HISTORY", "Treasury Yields · 1Y", "One-year co-movement and steepening across key Treasury maturities.")
        history_fig = go.Figure()
        history_colors = {"3M":"#f0a13a","5Y":"#2987cf","10Y":"#a4a9b0","30Y":"#35a85d"}
        for maturity,ticker in YIELD_TICKERS.items():
            if ticker not in prices:
                continue
            s=prices[ticker].dropna().tail(252)
            if s.empty:
                continue
            history_fig.add_trace(go.Scatter(x=s.index,y=s,name=maturity,line=dict(color=history_colors[maturity],width=1.8),
                hovertemplate=f"<b>UST {maturity}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}%<extra></extra>"))
        base_layout(history_fig,420,"x unified")
        history_fig.update_yaxes(title="Yield %",ticksuffix="%")
        st.plotly_chart(history_fig,width="stretch",config={"displaylogo":False})

    section_header("LEVEL & CHANGE", "Treasury Yields · Level & Change", "Red indicates falling yields; green indicates rising yields.")
    if yield_df.empty:
        st.warning("Treasury yield-index data was unavailable in this refresh.")
    else:
        yield_style=yield_df.style.format({"Yield":"{:.3f}%","1D (bps)":"{:+.1f}","1W (bps)":"{:+.1f}","1M (bps)":"{:+.1f}","YTD (bps)":"{:+.1f}"},na_rep="—").map(
            color_return,subset=["1D (bps)","1W (bps)","1M (bps)","YTD (bps)"])
        st.dataframe(yield_style,width="stretch",hide_index=True,height=220)

with tab_xray:
    section_header("LOOK-THROUGH RISK","ETF X-RAY · Look-Through Leverage Exposure",
                   "Issuer holdings are preferred. Inputs are portfolio capital weights; nominal exposure excludes leverage and effective beta exposure applies each ETF multiplier.")
    allocator=st.columns(4)
    allocation={
        "TQQQ":allocator[0].number_input("TQQQ · 3x Nasdaq (%)",0.0,100.0,30.0,1.0,key="xray_tqqq"),
        "SOXL":allocator[1].number_input("SOXL · 3x Semiconductors (%)",0.0,100.0,20.0,1.0,key="xray_soxl"),
        "QQQ":allocator[2].number_input("QQQ · Nasdaq Core (%)",0.0,100.0,30.0,1.0,key="xray_qqq"),
        "CASH":allocator[3].number_input("Cash (%)",0.0,100.0,20.0,1.0,key="xray_cash"),
    }
    allocation_sum=sum(allocation.values())
    if abs(allocation_sum-100)>.01:
        st.warning(f"Weights total {allocation_sum:.1f}%. Adjust to 100%; calculations below use the entered values without implicit normalization.")
    qqq_holdings,qqq_source=load_etf_holdings("QQQ"); soxx_holdings,soxx_source=load_etf_holdings("SOXX")
    exposure_rows=[]
    for portfolio_etf,lookthrough,leverage in (("TQQQ",qqq_holdings,3.0),("SOXL",soxx_holdings,3.0),("QQQ",qqq_holdings,1.0)):
        portfolio_weight=allocation[portfolio_etf]/100
        for _,holding in lookthrough.iterrows():
            constituent_weight=float(holding["Weight"])/100
            exposure_rows.append({"Ticker":holding["Ticker"],"Name":holding["Name"],
                "Nominal":portfolio_weight*constituent_weight,"Effective":portfolio_weight*leverage*constituent_weight,
                "Source ETF":portfolio_etf})
    exposure=pd.DataFrame(exposure_rows)
    if exposure.empty:
        st.warning("Underlying holdings are currently unavailable.")
    else:
        aggregate=exposure.groupby(["Ticker","Name"],as_index=False)[["Nominal","Effective"]].sum()
        aggregate["Nominal Weight (%)"]=aggregate["Nominal"]*100
        aggregate["Effective Beta Exposure (%)"]=aggregate["Effective"]*100
        aggregate["Concentration Risk"]=np.where(aggregate["Effective Beta Exposure (%)"]>25,"⚠️ Excessive single-name leverage","🟢 Controlled")
        aggregate=aggregate.sort_values("Effective Beta Exposure (%)",ascending=False)
        gross_beta=(allocation["TQQQ"]*3+allocation["SOXL"]*3+allocation["QQQ"])/100
        top_effective=float(aggregate["Effective Beta Exposure (%)"].iloc[0]); top_name=str(aggregate["Ticker"].iloc[0])
        xray_cards=st.columns(4)
        xray_cards[0].metric("PORTFOLIO ALLOCATION",f"{allocation_sum:.1f}%",delta_color="off")
        xray_cards[1].metric("GROSS BETA EXPOSURE",f"{gross_beta:.2f}x",delta_color="off")
        xray_cards[2].metric("TOP LOOK-THROUGH",top_name,f"{top_effective:.2f}% effective",delta_color="off")
        xray_cards[3].metric("CONCENTRATION LOCK","🚨 HIGH" if top_effective>25 else "🟢 CONTROLLED",delta_color="off")
        table_col,chart_col=st.columns([1.0,1.15],gap="medium")
        with table_col:
            section_header("CORE HOLDINGS","Top Look-Through Equity Exposures","Exposure to the same stock is aggregated across QQQ and semiconductor baskets.")
            output=aggregate.head(10)[["Ticker","Name","Nominal Weight (%)","Effective Beta Exposure (%)","Concentration Risk"]]
            style=output.style.format({"Nominal Weight (%)":"{:.2f}%","Effective Beta Exposure (%)":"{:.2f}%"}).map(
                lambda value:"color:#ff5874;font-weight:700" if str(value).startswith("⚠️") else "color:#31d6a0",subset=["Concentration Risk"])
            st.dataframe(style,width="stretch",hide_index=True,height=430)
        with chart_col:
            section_header("LEVERAGE MAP","Nominal vs Effective Exposure","Orange shows look-through invested capital; cyan shows effective beta exposure after leverage.")
            top=aggregate.head(10).sort_values("Effective Beta Exposure (%)")
            xray_fig=go.Figure()
            xray_fig.add_trace(go.Bar(y=top["Ticker"],x=top["Nominal Weight (%)"],orientation="h",name="Nominal Weight",marker_color="#f0a13a"))
            xray_fig.add_trace(go.Bar(y=top["Ticker"],x=top["Effective Beta Exposure (%)"],orientation="h",name="Effective Beta",marker_color="#35d5ff"))
            xray_fig.add_vline(x=25,line_dash="dash",line_color="#ff5874",annotation_text="25% CONCENTRATION")
            base_layout(xray_fig,430,"y unified"); xray_fig.update_layout(barmode="group"); xray_fig.update_xaxes(title="Portfolio Exposure (%)")
            st.plotly_chart(xray_fig,width="stretch",config={"displaylogo":False})
        st.caption(f"QQQ holdings source: {qqq_source}. SOXL look-through uses SOXX holdings from {soxx_source}. Issuer failures are explicitly labeled as built-in references and never presented as live holdings.")

with tab_deep:
    section_header("MACRO FINANCIAL CONDITIONS","Macro Deep Dive · FRED Financial Stress & Inflation",
                   "Direct FRED / OFR public CSV access without an API key, with a dated bundled snapshot as fallback.")
    fred_ids=("NFCI","OFRFSI","PCETRIM12M159SFRBDAL","PCETRIM1M158SFRBDAL","PCEPILFE","PCEPI","PI","PCE")
    fred,fred_status=load_fred_series(fred_ids)
    offline_ids=[series_id for series_id,source in fred_status.items() if source=="offline"]
    unavailable_ids=[series_id for series_id,source in fred_status.items() if source=="unavailable"]
    if offline_ids:
        offline_dates=[f"{series_id} {fred[series_id].dropna().index[-1]:%Y-%m-%d}" for series_id in offline_ids]
        st.caption("Offline fallback (not live): "+" · ".join(offline_dates)+". New official values take priority when connectivity returns.")
    if unavailable_ids:
        st.info("Unavailable in this refresh: "+", ".join(unavailable_ids)+".")
    macro_left,macro_right=st.columns(2,gap="medium")
    with macro_left:
        nfci=fred["NFCI"].dropna() if "NFCI" in fred else pd.Series(dtype=float)
        latest_nfci=float(nfci.iloc[-1]) if not nfci.empty else np.nan
        section_header("FINANCIAL CONDITIONS",f"NFCI · Current {'N/A' if not np.isfinite(latest_nfci) else f'{latest_nfci:+.3f}'}",
                       "Zero is historically neutral; negative values indicate looser conditions and positive values tighter conditions.")
        if nfci.empty: st.info("NFCI is currently unavailable.")
        else:
            view=nfci[nfci.index>=nfci.index[-1]-pd.Timedelta(days=365)]
            nfci_fig=go.Figure(go.Scatter(x=view.index,y=view,mode="lines",name="NFCI",fill="tozeroy",
                line=dict(color="#31d6a0" if latest_nfci<0 else "#ff5874",width=2.2)))
            nfci_fig.add_hline(y=0,line_dash="dash",line_color="#9aa8b9",annotation_text="NEUTRAL")
            base_layout(nfci_fig,420,"x unified"); nfci_fig.update_yaxes(title="NFCI")
            st.plotly_chart(nfci_fig,width="stretch",config={"displaylogo":False})
    with macro_right:
        ofr=fred["OFRFSI"].dropna() if "OFRFSI" in fred else pd.Series(dtype=float)
        latest_ofr=float(ofr.iloc[-1]) if not ofr.empty else np.nan
        section_header("SYSTEMIC STRESS",f"OFR Financial Stress Index · Current {'N/A' if not np.isfinite(latest_ofr) else f'{latest_ofr:+.2f}'}",
                       "Zero is neutral stress; +2 is the tactical high-stress threshold used by this dashboard.")
        if ofr.empty: st.info("OFR FSI is currently unavailable.")
        else:
            view=ofr[ofr.index>=ofr.index[-1]-pd.Timedelta(days=365)]
            ofr_fig=go.Figure(go.Scatter(x=view.index,y=view,mode="lines",name="OFR FSI",fill="tozeroy",line=dict(color="#f0a13a",width=2.2)))
            ofr_fig.add_hline(y=0,line_dash="dash",line_color="#9aa8b9",annotation_text="NEUTRAL")
            ofr_fig.add_hline(y=2,line_dash="dot",line_color="#ff5874",annotation_text="EXTREME STRESS")
            base_layout(ofr_fig,420,"x unified"); ofr_fig.update_yaxes(title="OFR FSI")
            st.plotly_chart(ofr_fig,width="stretch",config={"displaylogo":False})
    section_header("INFLATION TAPE","Inflation · Last 12 Monthly Releases",
                   "Headline/Core are year-over-year price-index changes; Trimmed Mean uses Dallas Fed series; income and spending are month-over-month. Each row is colored independently.")
    inflation=inflation_release_table(fred)
    if inflation.empty:
        st.info("Monthly inflation, income and spending series are currently unavailable.")
    else:
        st.dataframe(heat_style_table(inflation),width="stretch",height=285)
        st.caption("Units: %. The 6M Trimmed Mean is compounded from the official 1M annualized series. FRED releases have monthly or weekly lags; each current value refers to its latest valid publication date.")

with tab_pulse:
    section_header("EVENT PULSE", "Breaking News & Market Data", "Headlines are sourced from Yahoo Finance; market impact is classified with reproducible price, volatility, credit and breadth rules.")
    signal_rows=[]
    spy=prices["SPY"].dropna(); spy200=spy.rolling(200,min_periods=200).mean().iloc[-1]
    spy_dist=(spy.iloc[-1]/spy200-1)*100
    signal_rows.append({"Indicator":"SPY vs 200DMA","Current":f"{spy_dist:+.2f}%","Signal":"Supportive" if spy_dist>0 else "Adverse",
                        "Score":1 if spy_dist>0 else -1,"Interpretation":"Index is above its long-term trend" if spy_dist>0 else "Index is below its long-term trend"})
    if "^VIX" in prices and not prices["^VIX"].dropna().empty:
        vix=prices["^VIX"].dropna(); vix_now=float(vix.iloc[-1]); vix_day=last_return(vix,1)
        vix_score=1 if vix_now<20 and vix_day<=0 else (-1 if vix_now>25 or vix_day>8 else 0)
        signal_rows.append({"Indicator":"VIX","Current":f"{vix_now:.2f} ({vix_day:+.1f}% 1D)","Signal":{1:"Supportive",0:"Neutral",-1:"Adverse"}[vix_score],
                            "Score":vix_score,"Interpretation":"Low / declining volatility" if vix_score==1 else ("Volatility pressure is rising" if vix_score==-1 else "Volatility is in the middle range")})
    if all(t in prices for t in ["HYG","LQD"]):
        credit=prices["HYG"].div(prices["LQD"]); change=last_return(credit,20); score=1 if change>.3 else (-1 if change<-.3 else 0)
        signal_rows.append({"Indicator":"HYG / LQD Credit","Current":f"{change:+.2f}% 1M","Signal":{1:"Supportive",0:"Neutral",-1:"Adverse"}[score],"Score":score,
                            "Interpretation":"Credit risk appetite is improving" if score==1 else ("Credit risk appetite is weakening" if score==-1 else "Credit conditions are stable")})
    small_ratio=prices["IWM"].div(prices["SPY"]); small_change=last_return(small_ratio,20); score=1 if small_change>.5 else (-1 if small_change<-.5 else 0)
    signal_rows.append({"Indicator":"IWM / SPY Breadth","Current":f"{small_change:+.2f}% 1M","Signal":{1:"Supportive",0:"Neutral",-1:"Adverse"}[score],"Score":score,
                        "Interpretation":"Small-cap participation is broadening" if score==1 else ("Leadership is concentrated in large caps" if score==-1 else "Small and large caps are moving similarly")})
    breadth_net=(sp_adv-sp_dec)/500*100 if sp_missing<500 else np.nan
    score=1 if np.isfinite(breadth_net) and breadth_net>15 else (-1 if np.isfinite(breadth_net) and breadth_net<-15 else 0)
    signal_rows.append({"Indicator":"S&P 500 Company Breadth","Current":"N/A" if not np.isfinite(breadth_net) else f"{breadth_net:+.1f}% net",
                        "Signal":{1:"Supportive",0:"Neutral",-1:"Adverse"}[score],"Score":score,
                        "Interpretation":"Advancers clearly dominate" if score==1 else ("Decliners clearly dominate" if score==-1 else "Internals are balanced or incomplete")})
    if all(t in prices for t in ["CPER","GLD"]):
        cyclical=prices["CPER"].div(prices["GLD"]); change=last_return(cyclical,20); score=1 if change>1 else (-1 if change<-1 else 0)
        signal_rows.append({"Indicator":"Copper / Gold","Current":f"{change:+.2f}% 1M","Signal":{1:"Supportive",0:"Neutral",-1:"Adverse"}[score],"Score":score,
                            "Interpretation":"Cyclical expectations are improving" if score==1 else ("Defensive assets are outperforming" if score==-1 else "Macro cycle signal is neutral")})
    if "UUP" in prices:
        dollar_change=last_return(prices["UUP"],20); score=1 if dollar_change<-.7 else (-1 if dollar_change>.7 else 0)
        signal_rows.append({"Indicator":"U.S. Dollar · UUP","Current":f"{dollar_change:+.2f}% 1M","Signal":{1:"Supportive",0:"Neutral",-1:"Adverse"}[score],"Score":score,
                            "Interpretation":"A softer dollar eases financial conditions" if score==1 else ("Dollar strength pressures risk assets" if score==-1 else "Dollar movement is moderate")})
    signal_df=pd.DataFrame(signal_rows)
    composite=int(signal_df["Score"].sum()); max_score=max(1,len(signal_df)); normalized=composite/max_score*100
    regime="RISK-ON · SUPPORTIVE" if normalized>=25 else ("RISK-OFF · ADVERSE" if normalized<=-25 else "MIXED · NEUTRAL")
    pulse_cards=st.columns(4)
    pulse_cards[0].metric("COMPOSITE REGIME",regime,delta_color="off")
    pulse_cards[1].metric("SIGNAL SCORE",f"{composite:+d} / {max_score}",delta_color="off")
    pulse_cards[2].metric("SPY BREADTH",f"{sp_adv} A · {sp_dec} D",f"{sp_missing} missing",delta_color="off")
    pulse_cards[3].metric("VIX",f"{prices['^VIX'].dropna().iloc[-1]:.2f}" if "^VIX" in prices and not prices["^VIX"].dropna().empty else "N/A",delta_color="off")
    data_col,news_col=st.columns([.9,1.1],gap="medium")
    with data_col:
        section_header("DATA IMPACT", "Market Impact Matrix", "Green is supportive, red is adverse and gray is neutral. Rules describe the current data state only.")
        impact_style=signal_df.drop(columns=["Score"]).style.map(
            lambda v:"color:#31d6a0;font-weight:700" if v=="Supportive" else ("color:#ff5874;font-weight:700" if v=="Adverse" else "color:#9aa8b9"),subset=["Signal"])
        st.dataframe(impact_style,width="stretch",hide_index=True,height=430)
    with news_col:
        section_header("BREAKING TAPE", "Market-Moving Headlines", "Aggregates recent SPY, QQQ, VIX and large-cap technology headlines. Open a headline to view the source.")
        try:
            news_df=load_market_news(("SPY","QQQ","^VIX","NVDA"),7)
        except Exception:
            news_df=pd.DataFrame()
        if news_df.empty:
            st.info("Yahoo Finance returned no headlines in this refresh. Try again later.")
        else:
            st.dataframe(news_df[["Time","Source","Headline","URL"]],width="stretch",hide_index=True,height=430,
                column_config={"URL":st.column_config.LinkColumn("Open",display_text="↗ Read")})
    st.caption("Automated interpretation uses market proxies rather than a natural-language sentiment model. Read the source and verify publication time for breaking news.")

st.caption(f"DATA {latest_date:%Y-%m-%d} · SOURCE YAHOO FINANCE · LIVE PULSE CACHE 45S · RESEARCH CACHE 15M · STATIC SECTOR / CONSTITUENT WEIGHTS ARE VISUAL PROXIES · FOR RESEARCH ONLY, NOT INVESTMENT ADVICE")
