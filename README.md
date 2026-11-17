# Macro Quant Desk

**English-language cross-asset research terminal built with Streamlit and Plotly.**

[GitHub repository](https://github.com/doggy-ferry/macro-quant-desk-portfolio)

Macro Quant Desk converts a personal market-monitoring workflow into a public portfolio project without replacing the original terminal with a simplified landing page. The application keeps the dense information architecture of the working research system and translates the interface directly into English.

It is a decision-support tool, not an autonomous trading system, a forecasting model, or investment advice.

## Research workflow

- **00 Briefing** — live market pulse, verified primary-market flow, RRG upgrades, macro firewall and active alerts
- **01 Sector & Themes** — price RRG, issuer-share flow RRG, relative strength, tactical participation and sector regime matrix
- **02 Markets & Breadth** — index trends, exact breadth accounting, leadership and sector contribution
- **03 Heat Maps** — constituent performance and secondary-market activity
- **04 Macro & Commodities** — cross-asset returns and mean-reversion extremes
- **05 Technicals** — relative strength, MA regimes, SMA slopes, CAPM attribution, multi-timeframe indicators and continuation diagnostics
- **06 Regime & Pattern** — ADX regime engine and geometric pattern recognition
- **07 Primary ETF Flow** — issuer-share creation/redemption for supported ETFs
- **08 Circuit Breakers** — 200-day trend locks, credit risk appetite and VIX term structure
- **09 Fixed Income** — Treasury curve snapshots, history, levels and changes
- **10 ETF X-Ray** — look-through nominal and leveraged equity exposure
- **11 Macro Deep Dive** — FRED financial conditions, systemic stress and inflation releases
- **12 Event Pulse** — market-impact matrix and market-moving headlines

Option-chain, gamma and unusual-options modules are intentionally excluded from this public English edition. They are not fetched in the active application path.

## Architecture

```text
Yahoo Finance / FRED / issuer-share observations
                       |
                       v
       cached acquisition (45s / 15m / 1h)
                       |
                       v
 validation, normalization and missing-data controls
                       |
                       v
 breadth / RRG / trend / flow / macro / risk engines
                       |
                       v
          Streamlit terminal + Plotly charts
```

The main application is coordinated by `app.py`. Supporting engines cover dual RRG, primary-market flow, tactical participation, technical indicators, trend attribution and squeeze diagnostics. The visual system is centralized in `ui_theme.py`.

## Data principles

- Adjusted price and volume history comes primarily from Yahoo Finance through `yfinance`.
- Macro and financial-stress series use public FRED CSV endpoints with a bundled offline fallback.
- ETF primary-market flow is calculated only from consecutive issuer shares-outstanding observations: `ΔShares × Prior NAV`.
- Secondary-market dollar volume is never relabeled as ETF fund flow.
- Missing or stale observations remain explicit rather than being replaced with synthetic values.
- Free data sources may be delayed, rate-limited or revised.

## Local setup

Python 3.11 or 3.12 is recommended.

```powershell
git clone https://github.com/doggy-ferry/macro-quant-desk-portfolio.git
cd macro-quant-desk-portfolio
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

On Windows, `start_dashboard.cmd` provides a one-click launcher after the environment is installed.

## Configuration and security

The repository contains no production password or webhook credential. Optional private deployments read `AUTH_PASSWORD`, webhook URLs and Telegram credentials from Streamlit Secrets or environment variables. Local `.env`, `.streamlit/secrets.toml`, caches and virtual environments are ignored by Git.

## Validation

```powershell
.venv\Scripts\python.exe -m py_compile app.py ui_theme.py technical_indicators.py
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover technical calculations, squeeze detection, empty and incomplete datasets, date discontinuities, zero denominators and provider failures.

## Ownership and AI assistance

Macro Quant Desk is a self-directed project designed and implemented by **Xuanlin Yu**. I owned the market questions, feature selection, data-quality rules, validation criteria, architecture and final product decisions. AI-assisted tools supported implementation, refactoring, debugging, testing and documentation.

### Resume bullet

- Built and deployed an English-language Streamlit cross-asset research terminal integrating breadth, relative rotation, verified ETF creation/redemption, technical regimes, fixed income, macro stress and portfolio look-through analytics with cached data pipelines and explicit missing-data controls.
