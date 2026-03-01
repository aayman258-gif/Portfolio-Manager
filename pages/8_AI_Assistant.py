"""
Regime-Aware Portfolio Manager
Section 8: AI Assistant
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from openai import OpenAI

# Add parent directory to path (same pattern as other pages)
sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout

from calculations.optimizer import RegimeAwareOptimizer
from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader


# ─── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
)
apply_carbon_theme()

PLOTLY_TEMPLATE = "plotly_dark"  # kept for reference; charts use carbon_plotly_layout


# ─── System prompt / action contract ────────────────────────────────────────────

SYSTEM_PROMPT = """You are the in-app AI assistant for a local Streamlit portfolio manager.

You help the user understand and act on:
- Portfolio holdings, allocation, and P&L context
- Market regime detection outputs (Low Vol / High Vol / Trending / Mean Reversion) and their implications
- Portfolio optimization trade-offs and method selection
- Rebalancing suggestions and trade plans
- New position ideas (tickers + rationale) that the app will validate with live market data
- Options strategy suggestions for existing positions (educational; user decides)

Hard rules:
- Use ONLY the provided CONTEXT SNAPSHOT for portfolio-specific facts and numbers. If data is missing, say so clearly and ask for it.
- Do NOT invent tickers that are not publicly traded. Limit new position suggestions to 6 or fewer tickers.
- When uncertain, ask clarifying questions rather than guessing.
- Be concise and structured; use bullet points and numbered steps when helpful.
- Always include a brief "Not financial advice." reminder when making recommendations.

Action output:
If (and ONLY if) you want the app to perform an action, append exactly one action block at the very end of your message.

Allowed actions:

1) RUN_OPTIMIZATION
   Use when the user explicitly asks to run an optimization.
   Required field: "optimization_method" — one of: "max_sharpe", "min_volatility", "max_quadratic_utility"
   Optional field: "notes" (string)

2) SUGGEST_NEW_POSITIONS
   Use when the user asks for new tickers to consider adding.
   Required field: "suggestions" — list of objects, each with:
     - "ticker": string (e.g. "NVDA")
     - "rationale": string (1–3 sentences)
     - "sector": string (optional)

Format the action block EXACTLY as one of:

```json
{"type":"RUN_OPTIMIZATION","optimization_method":"max_sharpe","notes":"Optimizing for maximum risk-adjusted return."}
```

```json
{"type":"SUGGEST_NEW_POSITIONS","suggestions":[{"ticker":"NVDA","rationale":"Strong AI infrastructure tailwinds with 6-month momentum.","sector":"Technology"}]}
```

Do not include any other JSON code blocks in your response besides the single action block.
"""

# ─── Session state keys ──────────────────────────────────────────────────────────

CHAT_KEY = "ai_chat_messages"
LAST_ACTION_KEY = "ai_last_action_result"
MODEL_KEY = "ai_model"

AVAILABLE_MODELS = [
    "openai/gpt-5.2",
    "openai/gpt-5",
    "openai/o4-mini",
    "openai/o3-mini",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
]
DEFAULT_MODEL = "openai/gpt-5.2"


# ─── OpenRouter client ───────────────────────────────────────────────────────────

def _get_openrouter_api_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("OPENROUTER_API_KEY")


@st.cache_resource(show_spinner=False)
def _get_llm_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Streamlit Portfolio Manager",
        },
    )


def _stream_chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> Iterable[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for event in stream:
        try:
            delta = event.choices[0].delta
            chunk = getattr(delta, "content", None)
        except Exception:
            chunk = None
        if chunk:
            yield chunk


# ─── Regime detection (cached) ──────────────────────────────────────────────────

@st.cache_data(ttl=60 * 30, show_spinner=False)
def _compute_current_regime() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Fetch market data and classify current regime. Cached for 30 minutes."""
    loader = MarketDataLoader()
    detector = RegimeDetector()

    spy_data = loader.load_index_data("SPY", "2y")
    vix_data = loader.load_vix_data("2y")
    spy_prices, vix_prices = loader.align_data(spy_data, vix_data)

    regime_series, signals = detector.classify_regime(spy_prices, vix_prices)
    if regime_series is None or len(regime_series) == 0:
        return None, signals

    return str(regime_series.iloc[-1]), signals


# ─── Context snapshot builder ────────────────────────────────────────────────────

def _safe_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _build_portfolio_summary() -> Dict[str, Any]:
    ss = st.session_state

    positions_df: Optional[pd.DataFrame] = ss.get("positions", None)
    current_prices: Dict = ss.get("current_prices", {}) or {}
    total_value = _safe_float(ss.get("total_value", None))
    manual_positions = ss.get("manual_positions", None)
    optimization_result = ss.get("optimization_result", None)

    tickers: List[str] = []
    top_positions: List[Dict[str, Any]] = []

    if isinstance(positions_df, pd.DataFrame) and len(positions_df) > 0:
        df = positions_df.copy()
        for col in ["ticker", "shares", "avg_cost"]:
            if col not in df.columns:
                df[col] = None

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["price"] = df["ticker"].map(lambda t: _safe_float(current_prices.get(t)))
        df["position_value_est"] = (
            pd.to_numeric(df["shares"], errors="coerce")
            * pd.to_numeric(df["price"], errors="coerce")
        )
        df["weight_est"] = (
            df["position_value_est"] / total_value
            if total_value and total_value > 0
            else None
        )

        tickers = sorted(
            t for t in df["ticker"].dropna().unique().tolist()
            if t and t not in ("NAN", "NONE")
        )

        sort_col = "position_value_est" if df["position_value_est"].notna().any() else "shares"
        for _, row in df.sort_values(sort_col, ascending=False).head(10).iterrows():
            top_positions.append({
                "ticker": row.get("ticker"),
                "shares": _safe_float(row.get("shares")),
                "avg_cost": _safe_float(row.get("avg_cost")),
                "price": _safe_float(row.get("price")),
                "position_value_est": _safe_float(row.get("position_value_est")),
                "weight_est": _safe_float(row.get("weight_est")),
            })

    return {
        "tickers": tickers,
        "total_value": total_value,
        "top_positions_by_value": top_positions,
        "manual_positions_count": len(manual_positions) if isinstance(manual_positions, list) else None,
        "has_current_prices": bool(current_prices),
        "optimization_result": optimization_result,
    }


def _build_context_snapshot() -> Dict[str, Any]:
    try:
        regime, signals = _compute_current_regime()
        regime_error = None
    except Exception as e:
        regime, signals, regime_error = None, None, f"{type(e).__name__}: {e}"

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "portfolio": _build_portfolio_summary(),
        "market_regime": {
            "current_regime": regime,
            "signals": signals,
            "error": regime_error,
        },
        "app_capabilities": {
            "optimizer_methods": ["max_sharpe", "min_volatility", "max_quadratic_utility"],
            "data_source": "yfinance (free)",
        },
    }


# ─── Action parsing ──────────────────────────────────────────────────────────────

@dataclass
class ParsedAction:
    type: str
    payload: Dict[str, Any]


_ACTION_BLOCK_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)


def _parse_action(text: str) -> Optional[ParsedAction]:
    if not text:
        return None
    matches = _ACTION_BLOCK_RE.findall(text)
    if not matches:
        return None
    # Take the last JSON block (the action block, per contract)
    try:
        obj = json.loads(matches[-1].strip())
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    action_type = obj.get("type")
    if action_type not in {"RUN_OPTIMIZATION", "SUGGEST_NEW_POSITIONS"}:
        return None
    return ParsedAction(type=str(action_type), payload=obj)


# ─── Action: run optimization ────────────────────────────────────────────────────

def _get_tickers_from_session() -> List[str]:
    df = st.session_state.get("positions", None)
    if not isinstance(df, pd.DataFrame) or "ticker" not in df.columns:
        return []
    return sorted(
        t for t in df["ticker"].astype(str).str.upper().unique()
        if t and t not in ("NAN", "NONE", "")
    )


def _execute_run_optimization(action: ParsedAction) -> Dict[str, Any]:
    method = str(action.payload.get("optimization_method", "")).strip()
    if method not in {"max_sharpe", "min_volatility", "max_quadratic_utility"}:
        raise ValueError(f"Invalid optimization_method '{method}'. Must be one of: max_sharpe, min_volatility, max_quadratic_utility")

    tickers = _get_tickers_from_session()
    if not tickers:
        raise ValueError("No positions loaded. Please go to the Home page and load a portfolio first.")

    regime, _ = _compute_current_regime()
    if not regime:
        raise ValueError("Could not determine current market regime. Check your market data connection.")

    optimizer = RegimeAwareOptimizer()
    result = optimizer.optimize_portfolio(
        tickers=tickers,
        current_regime=regime,
        optimization_method=method,
    )

    # Persist under the existing session state key so other pages see it
    st.session_state["optimization_result"] = result

    return {
        "type": "RUN_OPTIMIZATION",
        "optimization_method": method,
        "tickers": tickers,
        "current_regime": regime,
        "result": result,
    }


def _render_optimizer_result(result: Dict[str, Any]) -> None:
    st.subheader("Optimization Result")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        v = result.get("expected_return")
        st.metric("Expected Return", f"{v:.2f}%" if v is not None else "—")
    with col2:
        v = result.get("volatility")
        st.metric("Volatility", f"{v:.2f}%" if v is not None else "—")
    with col3:
        v = result.get("sharpe_ratio")
        st.metric("Sharpe Ratio", f"{v:.3f}" if v is not None else "—")
    with col4:
        st.metric("Regime", result.get("regime", "—"))

    weights = result.get("weights", {})
    if isinstance(weights, dict) and weights:
        wdf = (
            pd.DataFrame([{"Ticker": k, "Weight": f"{v*100:.1f}%", "weight_raw": v}
                          for k, v in weights.items()])
            .sort_values("weight_raw", ascending=False)
        )

        st.markdown("**Target Weights**")
        st.dataframe(wdf[["Ticker", "Weight"]], use_container_width=True, hide_index=True)

        fig = px.bar(
            wdf,
            x="Ticker",
            y="weight_raw",
            title="Target Allocation",
            labels={"weight_raw": "Weight", "Ticker": ""},
        )
        fig.update_layout(**carbon_plotly_layout())
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        cash = result.get("cash_allocation")
        if cash is not None:
            st.caption(f"Cash allocation: {cash:.1f}% | Max equity allowed by regime: {result.get('max_equity_allowed', '—')}%")
    else:
        st.info("No weights returned by optimizer.")


# ─── Action: suggest new positions ──────────────────────────────────────────────

@st.cache_data(ttl=60 * 60, show_spinner=False)
def _validate_ticker(ticker: str) -> Dict[str, Any]:
    """Download 6 months of history and compute basic stats. Cached 1 hour."""
    t = ticker.strip().upper()
    if not t:
        return {"ticker": ticker, "valid": False, "error": "Empty ticker"}
    try:
        hist = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        if hist is None or hist.empty:
            return {"ticker": t, "valid": False, "error": "No price history returned"}

        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if "Close" not in hist.columns:
            return {"ticker": t, "valid": False, "error": "Close price column not found"}

        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
        ret_1d = ((last_close / prev_close) - 1.0) if prev_close else None
        first_close = float(hist["Close"].iloc[0]) if len(hist) >= 2 else None
        ret_6m = ((last_close / first_close) - 1.0) if first_close and first_close > 0 else None

        name, exchange, currency = None, None, None
        try:
            info = yf.Ticker(t).fast_info
            if info is not None:
                name = getattr(info, "shortName", None)
                exchange = getattr(info, "exchange", None)
                currency = getattr(info, "currency", None)
        except Exception:
            pass

        return {
            "ticker": t,
            "valid": True,
            "last_close": last_close,
            "ret_1d": ret_1d,
            "ret_6m": ret_6m,
            "name": name,
            "exchange": exchange,
            "currency": currency,
        }
    except Exception as e:
        return {"ticker": t, "valid": False, "error": f"{type(e).__name__}: {e}"}


def _execute_suggest_new_positions(action: ParsedAction) -> Dict[str, Any]:
    suggestions = action.payload.get("suggestions", [])
    if not isinstance(suggestions, list) or not suggestions:
        raise ValueError("SUGGEST_NEW_POSITIONS action must include a non-empty 'suggestions' list.")

    cleaned = []
    for s in suggestions[:6]:
        if not isinstance(s, dict):
            continue
        ticker = str(s.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        cleaned.append({
            "ticker": ticker,
            "rationale": str(s.get("rationale", "")).strip(),
            "sector": s.get("sector"),
        })

    if not cleaned:
        raise ValueError("No valid tickers found in suggestions.")

    validated = [
        {**s, "validation": _validate_ticker(s["ticker"])}
        for s in cleaned
    ]
    return {"type": "SUGGEST_NEW_POSITIONS", "suggestions": validated}


def _render_suggestion_cards(action_result: Dict[str, Any]) -> None:
    st.subheader("New Position Ideas — Validated")

    suggestions = action_result.get("suggestions", [])
    if not suggestions:
        st.info("No suggestions to display.")
        return

    for s in suggestions:
        ticker = s.get("ticker", "—")
        rationale = s.get("rationale", "")
        sector = s.get("sector")
        v = s.get("validation", {}) or {}

        valid = bool(v.get("valid", False))
        last_close = v.get("last_close")
        ret_1d = v.get("ret_1d")
        ret_6m = v.get("ret_6m")
        name = v.get("name")
        err = v.get("error")

        with st.container(border=True):
            cols = st.columns([1.5, 1, 1, 1])
            with cols[0]:
                st.markdown(f"### {ticker}")
                if name:
                    st.caption(name)
                if sector:
                    st.caption(f"Sector: {sector}")
            with cols[1]:
                st.metric("Data Valid", "✅ Yes" if valid else "❌ No")
            with cols[2]:
                st.metric("Last Close", f"${last_close:.2f}" if isinstance(last_close, float) else "—")
            with cols[3]:
                if isinstance(ret_1d, float):
                    st.metric("1D Return", f"{ret_1d*100:+.2f}%")
                else:
                    st.metric("1D Return", "—")

            if isinstance(ret_6m, float):
                st.caption(f"6-Month Return: {ret_6m*100:+.2f}%")

            if rationale:
                st.markdown(f"**Rationale:** {rationale}")

            if not valid and err:
                st.warning(f"Validation error: {err}")


# ─── Chat state helpers ──────────────────────────────────────────────────────────

def _init_chat() -> None:
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm your portfolio AI assistant. I can help you:\n\n"
                    "- **Understand** your portfolio, regime, scores, and optimization results\n"
                    "- **Run optimizations** -- just say: 'run a max Sharpe optimization'\n"
                    "- **Suggest new positions** -- ask: 'what new stocks should I consider?'\n"
                    "- **Explain options strategies** for your existing holdings\n\n"
                    "*Not financial advice. Load your portfolio on the Home page first for full context.*"
                ),
            },
        ]


def _clear_chat() -> None:
    for key in (CHAT_KEY, LAST_ACTION_KEY):
        st.session_state.pop(key, None)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## AI Assistant Controls")

        if MODEL_KEY not in st.session_state:
            st.session_state[MODEL_KEY] = DEFAULT_MODEL

        st.session_state[MODEL_KEY] = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state[MODEL_KEY])
            if st.session_state[MODEL_KEY] in AVAILABLE_MODELS
            else 0,
        )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            _clear_chat()
            st.rerun()

        st.divider()
        st.markdown("### Portfolio Context")

        positions_df = st.session_state.get("positions")
        current_prices = st.session_state.get("current_prices")
        total_value = st.session_state.get("total_value")
        opt_result = st.session_state.get("optimization_result")

        if isinstance(positions_df, pd.DataFrame) and len(positions_df) > 0:
            tickers = _get_tickers_from_session()
            st.success(f"Positions: {len(positions_df)} rows")
            st.caption(", ".join(tickers[:10]) + ("…" if len(tickers) > 10 else ""))
        else:
            st.warning("No positions loaded")

        if isinstance(current_prices, dict) and current_prices:
            st.success(f"Prices: {len(current_prices)} tickers")
        else:
            st.warning("No prices loaded")

        tv = _safe_float(total_value)
        if tv is not None:
            st.success(f"Portfolio value: ${tv:,.0f}")
        else:
            st.warning("Total value not set")

        if isinstance(opt_result, dict) and opt_result:
            st.success("Optimization result: present")
        else:
            st.info("No optimization result yet")

        try:
            regime, _ = _compute_current_regime()
            if regime:
                st.success(f"Regime: {regime}")
            else:
                st.warning("Regime: unavailable")
        except Exception:
            st.warning("Regime: error computing")

        st.divider()
        st.caption("API key read from st.secrets or OPENROUTER_API_KEY env var.")


# ─── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🤖 AI Portfolio Assistant")
    st.markdown("Ask me about your portfolio, regime, optimization, or new position ideas.")

    _render_sidebar()
    _init_chat()

    api_key = _get_openrouter_api_key()
    if not api_key:
        st.error(
            "Missing OpenRouter API key. "
            "Add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml` or set it as an environment variable."
        )
        st.stop()

    client = _get_llm_client(api_key=api_key)

    # Render existing conversation (skip system message)
    for msg in st.session_state[CHAT_KEY]:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Ask about your portfolio, regime, optimization, or new position ideas…")

    if not user_text:
        # Show last action result in an expander if available
        last_action = st.session_state.get(LAST_ACTION_KEY)
        if isinstance(last_action, dict) and last_action:
            with st.expander("Last action result", expanded=False):
                if last_action.get("type") == "RUN_OPTIMIZATION":
                    _render_optimizer_result(last_action.get("result", {}))
                elif last_action.get("type") == "SUGGEST_NEW_POSITIONS":
                    _render_suggestion_cards(last_action)
        return

    # Append user message and render it
    st.session_state[CHAT_KEY].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Build LLM message list:
    #   [system] + [fresh context snapshot] + [last 12 conversation turns]
    snapshot = _build_context_snapshot()
    context_msg = {
        "role": "user",
        "content": "CONTEXT SNAPSHOT (JSON):\n" + json.dumps(snapshot, indent=2, default=str),
    }

    non_system = [m for m in st.session_state[CHAT_KEY] if m["role"] != "system"]
    recent = non_system[-12:]

    llm_messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        context_msg,
        *[{"role": m["role"], "content": m["content"]} for m in recent],
    ]

    # Stream the response
    model = st.session_state[MODEL_KEY]
    assistant_text = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for chunk in _stream_chat_completion(client, model, llm_messages):
                assistant_text += chunk
                placeholder.markdown(assistant_text + "▌")
            placeholder.markdown(assistant_text)
        except Exception as e:
            st.error(f"LLM request failed: {type(e).__name__}: {e}")
            st.stop()

    st.session_state[CHAT_KEY].append({"role": "assistant", "content": assistant_text})

    # Parse and execute any action block in the response
    action = _parse_action(assistant_text)
    if action is None:
        return

    st.divider()
    st.markdown(f"**Action detected:** `{action.type}`")

    try:
        if action.type == "RUN_OPTIMIZATION":
            with st.spinner("Running portfolio optimization…"):
                action_result = _execute_run_optimization(action)
            st.session_state[LAST_ACTION_KEY] = action_result
            st.success("Optimization complete.")
            _render_optimizer_result(action_result["result"])

        elif action.type == "SUGGEST_NEW_POSITIONS":
            with st.spinner("Validating suggested tickers with yfinance…"):
                action_result = _execute_suggest_new_positions(action)
            st.session_state[LAST_ACTION_KEY] = action_result
            st.success("Tickers validated.")
            _render_suggestion_cards(action_result)

    except Exception as e:
        st.session_state[LAST_ACTION_KEY] = {"type": action.type, "error": str(e)}
        st.error(f"Action failed: {type(e).__name__}: {e}")
        st.info("Check that your portfolio is loaded on the Home page, then try again.")


main()
