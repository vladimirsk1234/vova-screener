#!/usr/bin/env python3
"""
PE/EPS vs Sequence Vova closed-trade analysis.

Builds closed trades from Mongo trackedSignals (if MONGO_URI) or close-ledger
replay over STOCK-TICKERS.txt, joins current Yahoo fundamentals (yfinance),
segments win/loss by PE/EPS, and simulates filters F1–F5 vs baseline.

Caveat: Yahoo .info is current-only (look-ahead vs historical entry dates).

Usage:
  python scripts/analyze_pe_eps_trades.py --limit 200
  python scripts/analyze_pe_eps_trades.py --resume
  python scripts/analyze_pe_eps_trades.py --trades-only   # skip fundamentals refresh
  python scripts/analyze_pe_eps_trades.py --report-only   # reuse cached trades+fundamentals
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from data_utils import fill_last_bar_ohlc, resample_to_timeframe  # noqa: E402
from ohlc_yahoo import (  # noqa: E402
    BATCH_PAUSE_SEC,
    CHUNK_SIZE,
    FETCH_PERIOD,
    INTERVAL,
    RATE_LIMIT_BACKOFF_SEC,
    RATE_LIMIT_COOLDOWN_SEC,
    REQUIRED_COLS,
    fetch_batch_dfs,
    is_rate_limit,
)
from sequence_vova import run_sequence_vova_close_ledger  # noqa: E402
from ticker_data import (  # noqa: E402
    TV_LIST_STOCK_TICKERS,
    _eps_from_yf_info,
    _float_field,
    read_list_file,
)

CACHE_DIR = ROOT / ".cache" / "pe_eps_analysis"
TRADES_CACHE = CACHE_DIR / "closed_trades.jsonl"
FUND_CACHE = CACHE_DIR / "fundamentals.json"
REPORT_CSV = ROOT / "reports" / "pe_eps_trade_analysis.csv"
REPORT_MD = ROOT / "reports" / "pe_eps_filter_summary.md"

TIMEFRAMES = ("Daily", "Weekly", "Monthly")
MIN_BARS = 50
DAILY_LOOKBACK_CALENDAR_DAYS = 365 * 2 + 30  # ~2y History window + cushion
LEDGER_OPTS = {
    "atr_len": 14,
    "min_rr": 1.5,
    "use_last_hl_sl": True,
    "risk_dollars": 100.0,
    "no_rr_req": False,
}
PE_RATE_LIMIT_PER_SEC = 4.0
MIN_TRADES_FOR_VERDICT = 30

FILTERS: dict[str, str] = {
    "F1": "trailingPE > 0",
    "F2": "trailingEps > 0",
    "F3": "trailingPE in [5, 25]",
    "F4": "forwardPE < trailingPE (both > 0)",
    "F5": "F1 ∧ F2",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite(val: Any) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_trades_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_trades_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _ticker_universe(limit: int | None) -> list[tuple[str, str, str]]:
    tickers, tv_map, name_map, err = read_list_file(TV_LIST_STOCK_TICKERS)
    if err:
        raise RuntimeError(err)
    out = [(tv_map.get(y) or y, y, name_map.get(y) or "") for y in tickers]
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def _try_mongo_closed_trades() -> list[dict] | None:
    uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
    if not uri:
        return None
    try:
        from pymongo import MongoClient
    except ImportError:
        print("MONGO_URI set but pymongo not installed; falling back to ledger replay")
        return None

    print(f"Loading closed trades from Mongo…")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db_name = os.environ.get("MONGO_DB", "vova")
    coll = client[db_name]["trackedSignals"]
    query = {"status": "closed", "universe": "Stocks"}
    cursor = coll.find(
        query,
        {
            "yahooTicker": 1,
            "symbol": 1,
            "companyName": 1,
            "tf": 1,
            "entry": 1,
            "exitPrice": 1,
            "openedAsOf": 1,
            "exitDate": 1,
            "rrAtEntry": 1,
            "pnlUsd": 1,
            "pnlR": 1,
            "pnlPct": 1,
            "holdPeriods": 1,
            "shares": 1,
            "riskUsd": 1,
        },
    )
    rows: list[dict] = []
    for doc in cursor:
        pnl = _finite(doc.get("pnlUsd"))
        if pnl is None:
            continue
        rows.append(
            {
                "source": "mongo",
                "yahooTicker": doc.get("yahooTicker") or "",
                "symbol": doc.get("symbol") or doc.get("yahooTicker") or "",
                "companyName": doc.get("companyName") or "",
                "tf": doc.get("tf") or "",
                "entry_date": str(doc.get("openedAsOf") or "")[:10],
                "exit_date": str(doc.get("exitDate") or "")[:10],
                "entry_price": _finite(doc.get("entry")),
                "exit_price": _finite(doc.get("exitPrice")),
                "entry_rr": _finite(doc.get("rrAtEntry")),
                "pnl_usd": pnl,
                "pnl_r": _finite(doc.get("pnlR")),
                "pnl_pct": _finite(doc.get("pnlPct")),
                "hold_periods": _finite(doc.get("holdPeriods")),
                "shares": _finite(doc.get("shares")),
                "risk_usd": _finite(doc.get("riskUsd")) or LEDGER_OPTS["risk_dollars"],
            }
        )
    print(f"Mongo closed Stocks trades: {len(rows)}")
    return rows


def _prepare_tf_frame(df_daily: pd.DataFrame, tf: str) -> pd.DataFrame | None:
    if df_daily is None or df_daily.empty:
        return None
    frame = df_daily.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    if tf == "Daily":
        cutoff = frame.index.max() - pd.Timedelta(days=DAILY_LOOKBACK_CALENDAR_DAYS)
        frame = frame.loc[frame.index >= cutoff]
    else:
        frame = resample_to_timeframe(frame, tf)
    if frame is None or frame.empty:
        return None
    frame = fill_last_bar_ohlc(frame)
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    if len(frame) < MIN_BARS:
        return None
    return frame


def _date_at(index: pd.DatetimeIndex, i: int) -> str:
    ts = index[i]
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]


def _ledger_closed_rows(
    yahoo: str,
    tv: str,
    name: str,
    df_daily: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    for tf in TIMEFRAMES:
        frame = _prepare_tf_frame(df_daily, tf)
        if frame is None:
            continue
        ledger = run_sequence_vova_close_ledger(frame, **LEDGER_OPTS) or []
        dates = frame.index
        for trade in ledger:
            exit_index = trade.get("exit_index")
            if exit_index is None:
                continue
            pnl = _finite(trade.get("pnl_dollars"))
            if pnl is None:
                continue
            entry_i = int(trade["entry_index"])
            exit_i = int(exit_index)
            entry_price = _finite(trade.get("entry_price"))
            entry_sl = _finite(trade.get("entry_sl"))
            pnl_r = _finite(trade.get("close_rr"))
            rows.append(
                {
                    "source": "ledger",
                    "yahooTicker": yahoo,
                    "symbol": tv,
                    "companyName": name,
                    "tf": tf,
                    "entry_date": _date_at(dates, entry_i),
                    "exit_date": _date_at(dates, exit_i),
                    "entry_price": entry_price,
                    "exit_price": _finite(trade.get("exit_price")),
                    "entry_rr": _finite(trade.get("entry_rr")),
                    "pnl_usd": pnl,
                    "pnl_r": pnl_r,
                    "pnl_pct": _finite(trade.get("pnl_pct")),
                    "hold_periods": float(exit_i - entry_i),
                    "shares": _finite(trade.get("position_size")),
                    "risk_usd": LEDGER_OPTS["risk_dollars"],
                    "entry_sl": entry_sl,
                }
            )
    return rows


def collect_ledger_trades(
    universe: list[tuple[str, str, str]],
    *,
    resume: bool,
) -> list[dict]:
    existing = _load_trades_jsonl(TRADES_CACHE) if resume else []
    if not resume and TRADES_CACHE.exists():
        TRADES_CACHE.unlink()
        existing = []

    done_tickers = {r["yahooTicker"] for r in existing if r.get("source") == "ledger"}
    pending = [(tv, y, n) for tv, y, n in universe if y not in done_tickers]
    print(
        f"Ledger replay: {len(pending)} pending "
        f"(cached tickers with trades file: {len(done_tickers)})"
    )

    consecutive_rate_limits = 0
    for batch_start in range(0, len(pending), CHUNK_SIZE):
        if batch_start > 0:
            time.sleep(BATCH_PAUSE_SEC)
        batch_entries = pending[batch_start : batch_start + CHUNK_SIZE]
        batch = [y for _, y, _ in batch_entries]
        ticker_dfs, rate_limited = fetch_batch_dfs(batch)

        if rate_limited:
            consecutive_rate_limits += 1
            if consecutive_rate_limits >= 2:
                print(f"  Cooling down {RATE_LIMIT_COOLDOWN_SEC:.0f}s…")
                time.sleep(RATE_LIMIT_COOLDOWN_SEC)
                consecutive_rate_limits = 0
                ticker_dfs, rate_limited = fetch_batch_dfs(batch)
        else:
            consecutive_rate_limits = 0

        batch_rows: list[dict] = []
        # Mark processed tickers even with zero closed trades so resume skips them.
        markers: list[dict] = []
        for tv, yahoo, name in batch_entries:
            df = ticker_dfs.get(yahoo)
            if df is None or getattr(df, "empty", True):
                markers.append(
                    {
                        "source": "ledger",
                        "yahooTicker": yahoo,
                        "symbol": tv,
                        "companyName": name,
                        "tf": "_processed",
                        "entry_date": "",
                        "exit_date": "",
                        "pnl_usd": None,
                        "skip_reason": "NO_DAILY" if not rate_limited else "RATE_LIMIT",
                    }
                )
                continue
            daily = df[REQUIRED_COLS].copy()
            if not isinstance(daily.index, pd.DatetimeIndex):
                daily.index = pd.to_datetime(daily.index)
            closed = _ledger_closed_rows(yahoo, tv, name, daily)
            if closed:
                batch_rows.extend(closed)
            else:
                markers.append(
                    {
                        "source": "ledger",
                        "yahooTicker": yahoo,
                        "symbol": tv,
                        "companyName": name,
                        "tf": "_processed",
                        "entry_date": "",
                        "exit_date": "",
                        "pnl_usd": None,
                        "skip_reason": "NO_CLOSED_TRADES",
                    }
                )

        _append_trades_jsonl(TRADES_CACHE, batch_rows + markers)
        existing.extend(batch_rows + markers)
        done = min(batch_start + CHUNK_SIZE, len(pending))
        closed_n = sum(1 for r in existing if r.get("pnl_usd") is not None)
        print(
            f"  OHLC+ledger {done}/{len(pending)} pending batch — "
            f"closed trades so far: {closed_n}"
        )

    return [r for r in existing if r.get("pnl_usd") is not None]


def fetch_fundamentals(tickers: list[str], *, resume: bool) -> dict[str, dict]:
    cache = _load_json(FUND_CACHE) if resume else {}
    checked: dict = cache.setdefault("checked", {})
    pending = [t for t in tickers if t not in checked]
    print(f"Fundamentals pending: {len(pending)} (cached: {len(tickers) - len(pending)})")

    min_interval = 1.0 / PE_RATE_LIMIT_PER_SEC
    last_call = 0.0
    for i, yahoo in enumerate(pending, 1):
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_call = time.monotonic()
        try:
            info = yf.Ticker(yahoo).info or {}
            te, fe = _eps_from_yf_info(info)
            checked[yahoo] = {
                "trailingPE": _float_field(info, "trailingPE"),
                "forwardPE": _float_field(info, "forwardPE"),
                "trailingEps": te,
                "forwardEps": fe,
                "marketCap": _float_field(info, "marketCap"),
                "quoteType": info.get("quoteType"),
                "fetchedAt": _utcnow_iso(),
                "reason": "OK",
            }
        except Exception as exc:
            reason = "RATE_LIMIT" if is_rate_limit(exc) else "INFO_ERROR"
            checked[yahoo] = {
                "trailingPE": None,
                "forwardPE": None,
                "trailingEps": None,
                "forwardEps": None,
                "marketCap": None,
                "quoteType": None,
                "fetchedAt": _utcnow_iso(),
                "reason": reason,
                "error": str(exc)[:200],
            }
            if reason == "RATE_LIMIT":
                print("  Rate limit on .info — sleep 20s")
                time.sleep(20.0)

        if i % 25 == 0 or i == len(pending):
            _save_json(FUND_CACHE, cache)
            print(f"  Fundamentals {i}/{len(pending)}")

    _save_json(FUND_CACHE, cache)
    return checked


def pe_bucket(pe: float | None) -> str:
    if pe is None:
        return "missing"
    if pe <= 0:
        return "<=0"
    if pe <= 15:
        return "(0,15]"
    if pe <= 25:
        return "(15,25]"
    if pe <= 40:
        return "(25,40]"
    return ">40"


def eps_bucket(eps: float | None) -> str:
    if eps is None:
        return "missing"
    if eps <= 0:
        return "<=0"
    return ">0"


def passes_filter(fid: str, fund: dict) -> bool:
    pe = _finite(fund.get("trailingPE"))
    fpe = _finite(fund.get("forwardPE"))
    te = _finite(fund.get("trailingEps"))
    if fid == "F1":
        return pe is not None and pe > 0
    if fid == "F2":
        return te is not None and te > 0
    if fid == "F3":
        return pe is not None and 5.0 <= pe <= 25.0
    if fid == "F4":
        return (
            pe is not None
            and fpe is not None
            and pe > 0
            and fpe > 0
            and fpe < pe
        )
    if fid == "F5":
        return passes_filter("F1", fund) and passes_filter("F2", fund)
    raise ValueError(fid)


def _metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winRatePct": None,
            "pnlUsd": 0.0,
            "avgR": None,
            "avgRrEntry": None,
            "pnlPerTrade": None,
        }
    wins = sum(1 for r in rows if (r.get("pnl_usd") or 0) > 0)
    losses = n - wins
    pnl = sum(float(r["pnl_usd"]) for r in rows)
    rs = [r["pnl_r"] for r in rows if _finite(r.get("pnl_r")) is not None]
    rrs = [r["entry_rr"] for r in rows if _finite(r.get("entry_rr")) is not None]
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / n, 2),
        "pnlUsd": round(pnl, 2),
        "avgR": round(float(np.mean(rs)), 3) if rs else None,
        "avgRrEntry": round(float(np.mean(rrs)), 3) if rrs else None,
        "pnlPerTrade": round(pnl / n, 2),
    }


def join_trades(trades: list[dict], funds: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for t in trades:
        yahoo = t["yahooTicker"]
        fund = funds.get(yahoo) or {}
        pe = _finite(fund.get("trailingPE"))
        te = _finite(fund.get("trailingEps"))
        fe = _finite(fund.get("forwardEps"))
        fpe = _finite(fund.get("forwardPE"))
        win = 1 if (t.get("pnl_usd") or 0) > 0 else 0
        row = {
            **t,
            "trailingPE": pe,
            "forwardPE": fpe,
            "trailingEps": te,
            "forwardEps": fe,
            "marketCap": _finite(fund.get("marketCap")),
            "fund_reason": fund.get("reason"),
            "pe_bucket": pe_bucket(pe),
            "trailing_eps_bucket": eps_bucket(te),
            "forward_eps_bucket": eps_bucket(fe),
            "win": win,
        }
        for fid in FILTERS:
            row[f"pass_{fid}"] = int(passes_filter(fid, fund))
        out.append(row)
    return out


def segment_table(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "missing")].append(r)
    table = []
    for bucket in sorted(groups.keys()):
        m = _metrics(groups[bucket])
        table.append({"bucket": bucket, **m})
    return table


def simulate_filters(rows: list[dict]) -> list[dict]:
    results: list[dict] = []
    by_tf: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tf[r["tf"]].append(r)
        by_tf["All"].append(r)

    for tf in ["All", *TIMEFRAMES]:
        base_rows = by_tf.get(tf) or []
        base = _metrics(base_rows)
        results.append(
            {
                "tf": tf,
                "filter": "baseline",
                "rule": "none",
                **base,
                "cutWins": 0,
                "cutLosses": 0,
                "deltaWinRatePct": 0.0,
                "deltaPnlUsd": 0.0,
                "deltaPnlPerTrade": 0.0,
                "improves": None,
                "verdict": "baseline",
            }
        )
        for fid, rule in FILTERS.items():
            kept = [r for r in base_rows if r.get(f"pass_{fid}")]
            cut = [r for r in base_rows if not r.get(f"pass_{fid}")]
            m = _metrics(kept)
            cut_wins = sum(1 for r in cut if r.get("win"))
            cut_losses = len(cut) - cut_wins
            d_wr = None
            if base["winRatePct"] is not None and m["winRatePct"] is not None:
                d_wr = round(m["winRatePct"] - base["winRatePct"], 2)
            d_pnl = round(m["pnlUsd"] - base["pnlUsd"], 2)
            d_ppt = None
            if base["pnlPerTrade"] is not None and m["pnlPerTrade"] is not None:
                d_ppt = round(m["pnlPerTrade"] - base["pnlPerTrade"], 2)

            improves = False
            verdict = "insufficient_data"
            if m["trades"] >= MIN_TRADES_FOR_VERDICT and base["trades"] >= MIN_TRADES_FOR_VERDICT:
                wr_up = d_wr is not None and d_wr > 0
                pnl_ok = m["pnlUsd"] >= base["pnlUsd"] or (
                    d_ppt is not None and d_ppt > 0 and m["pnlUsd"] >= 0.9 * base["pnlUsd"]
                )
                cut_skew = cut_losses > cut_wins
                improves = bool(wr_up and pnl_ok and cut_skew)
                if improves:
                    verdict = "improves"
                elif wr_up and not pnl_ok:
                    verdict = "wr_up_pnl_down"
                elif not wr_up and pnl_ok:
                    verdict = "pnl_ok_wr_flat_or_down"
                else:
                    verdict = "no_improve"

            results.append(
                {
                    "tf": tf,
                    "filter": fid,
                    "rule": rule,
                    **m,
                    "cutWins": cut_wins,
                    "cutLosses": cut_losses,
                    "deltaWinRatePct": d_wr,
                    "deltaPnlUsd": d_pnl,
                    "deltaPnlPerTrade": d_ppt,
                    "improves": improves if verdict != "insufficient_data" else None,
                    "verdict": verdict,
                }
            )
    return results


def write_trade_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "yahooTicker",
        "symbol",
        "companyName",
        "tf",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "entry_rr",
        "pnl_usd",
        "pnl_r",
        "pnl_pct",
        "hold_periods",
        "win",
        "trailingPE",
        "forwardPE",
        "trailingEps",
        "forwardEps",
        "marketCap",
        "pe_bucket",
        "trailing_eps_bucket",
        "forward_eps_bucket",
        "fund_reason",
        "pass_F1",
        "pass_F2",
        "pass_F3",
        "pass_F4",
        "pass_F5",
        "source",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}%"


def _fmt_num(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v}"


def write_summary_md(
    joined: list[dict],
    filter_rows: list[dict],
    pe_seg: list[dict],
    te_seg: list[dict],
    fe_seg: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_all = next(r for r in filter_rows if r["tf"] == "All" and r["filter"] == "baseline")
    lines: list[str] = []
    lines.append("# PE/EPS vs Sequence Vova closed trades")
    lines.append("")
    lines.append(f"Generated: {_utcnow_iso()}")
    lines.append("")
    lines.append("## Caveat (look-ahead)")
    lines.append("")
    lines.append(
        "Fundamentals come from **current** Yahoo Finance `.info` via yfinance "
        "(`trailingPE`, `forwardPE`, `trailingEps`, `forwardEps`). They are **not** "
        "point-in-time at trade entry. Results are a hypothesis for a live filter on "
        "**new** signals, not a strict historical backtest."
    )
    lines.append("")
    lines.append("## Data")
    lines.append("")
    sources = Counter(r.get("source") for r in joined)
    lines.append(f"- Closed trades: **{len(joined)}** ({dict(sources)})")
    lines.append(f"- Universe: Stocks only (`{TV_LIST_STOCK_TICKERS}`); ETFs excluded")
    lines.append(
        f"- Ledger params: `min_rr={LEDGER_OPTS['min_rr']}`, "
        f"`risk_dollars={LEDGER_OPTS['risk_dollars']}`, "
        f"`no_rr_req={LEDGER_OPTS['no_rr_req']}`"
    )
    lines.append(
        "- Windows: Daily ~2y daily bars; Weekly/Monthly resampled from 10y daily "
        f"(Yahoo `{FETCH_PERIOD}` / `{INTERVAL}`)"
    )
    lines.append(f"- Win definition: `pnl_usd > 0`")
    lines.append(
        f"- Improve criterion: win rate ↑ AND (net P&L ≥ baseline OR P&L/trade ↑ with "
        f"P&L ≥ 90% baseline) AND cut losses > cut wins; min {MIN_TRADES_FOR_VERDICT} trades"
    )
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(
        f"| TF | Trades | Win rate | Net P&L | Avg R | Avg RR entry |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tf in ["All", *TIMEFRAMES]:
        b = next(r for r in filter_rows if r["tf"] == tf and r["filter"] == "baseline")
        lines.append(
            f"| {tf} | {b['trades']} | {_fmt_pct(b['winRatePct'])} | "
            f"{b['pnlUsd']} | {_fmt_num(b['avgR'])} | {_fmt_num(b['avgRrEntry'])} |"
        )
    lines.append("")
    lines.append("## PE buckets (All TF)")
    lines.append("")
    lines.append("| Bucket | Trades | Wins | Win rate | Net P&L | Avg R |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in pe_seg:
        lines.append(
            f"| {row['bucket']} | {row['trades']} | {row['wins']} | "
            f"{_fmt_pct(row['winRatePct'])} | {row['pnlUsd']} | {_fmt_num(row['avgR'])} |"
        )
    lines.append("")
    lines.append("## Trailing EPS buckets (All TF)")
    lines.append("")
    lines.append("| Bucket | Trades | Wins | Win rate | Net P&L | Avg R |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in te_seg:
        lines.append(
            f"| {row['bucket']} | {row['trades']} | {row['wins']} | "
            f"{_fmt_pct(row['winRatePct'])} | {row['pnlUsd']} | {_fmt_num(row['avgR'])} |"
        )
    lines.append("")
    lines.append("## Forward EPS buckets (All TF)")
    lines.append("")
    lines.append("| Bucket | Trades | Wins | Win rate | Net P&L | Avg R |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in fe_seg:
        lines.append(
            f"| {row['bucket']} | {row['trades']} | {row['wins']} | "
            f"{_fmt_pct(row['winRatePct'])} | {row['pnlUsd']} | {_fmt_num(row['avgR'])} |"
        )
    lines.append("")
    lines.append("## Filter simulation vs baseline")
    lines.append("")
    lines.append(
        "| TF | Filter | Rule | Trades | Win rate | Δ WR | Net P&L | Δ P&L | "
        "P&L/trade | Cut wins | Cut losses | Verdict |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in filter_rows:
        if r["filter"] == "baseline":
            continue
        lines.append(
            f"| {r['tf']} | {r['filter']} | {r['rule']} | {r['trades']} | "
            f"{_fmt_pct(r['winRatePct'])} | {_fmt_num(r['deltaWinRatePct'])} | "
            f"{r['pnlUsd']} | {_fmt_num(r['deltaPnlUsd'])} | {_fmt_num(r['pnlPerTrade'])} | "
            f"{r['cutWins']} | {r['cutLosses']} | {r['verdict']} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    improves = [
        r
        for r in filter_rows
        if r["filter"] != "baseline" and r.get("verdict") == "improves"
    ]
    if not improves:
        lines.append(
            "**No PE/EPS filter met the improve criterion** on any timeframe with "
            f"≥{MIN_TRADES_FOR_VERDICT} trades. Do **not** wire a live PE/EPS gate yet."
        )
    else:
        lines.append("Filters that met the improve criterion:")
        lines.append("")
        for r in improves:
            lines.append(
                f"- **{r['filter']}** on {r['tf']}: WR {_fmt_pct(r['winRatePct'])} "
                f"(Δ {r['deltaWinRatePct']}), P&L {r['pnlUsd']} (Δ {r['deltaPnlUsd']}), "
                f"cut {r['cutLosses']} losses / {r['cutWins']} wins"
            )
        lines.append("")
        lines.append(
            "Even with uplift, persist PE/EPS at signal open before trusting History "
            "segmentation — current Yahoo values still embed look-ahead."
        )
    lines.append("")
    lines.append("### Source recommendation")
    lines.append("")
    lines.append(
        "**yfinance / Yahoo `.info` is the right free source** for this stack "
        "(already used for watermark and gap-scan `trailingPE > 0`). "
        "Do not add Finviz scraping for this analysis."
    )
    lines.append("")
    lines.append(f"Trade-level CSV: `{REPORT_CSV.relative_to(ROOT)}`")
    lines.append("")
    # Keep baseline referenced so unused-var lints stay quiet in editors.
    _ = base_all
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Max tickers from STOCK-TICKERS.txt")
    ap.add_argument("--resume", action="store_true", help="Resume trades + fundamentals caches")
    ap.add_argument("--report-only", action="store_true", help="Only rebuild reports from caches")
    ap.add_argument(
        "--trades-only",
        action="store_true",
        help="Collect trades then stop (no fundamentals / report)",
    )
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        trades = [r for r in _load_trades_jsonl(TRADES_CACHE) if r.get("pnl_usd") is not None]
        funds = _load_json(FUND_CACHE).get("checked") or {}
        if not trades:
            print("No cached closed trades; run without --report-only first")
            return 1
    else:
        mongo_trades = _try_mongo_closed_trades()
        if mongo_trades is not None:
            trades = mongo_trades
            # Refresh jsonl for report-only reuse
            if TRADES_CACHE.exists():
                TRADES_CACHE.unlink()
            _append_trades_jsonl(TRADES_CACHE, trades)
        else:
            universe = _ticker_universe(args.limit)
            print(f"Universe size: {len(universe)}")
            trades = collect_ledger_trades(universe, resume=args.resume)
        print(f"Closed trades collected: {len(trades)}")

        if args.trades_only:
            return 0

        tickers = sorted({t["yahooTicker"] for t in trades if t.get("yahooTicker")})
        funds = fetch_fundamentals(tickers, resume=args.resume)

    joined = join_trades(trades, funds)
    write_trade_csv(joined, REPORT_CSV)
    pe_seg = segment_table(joined, "pe_bucket")
    te_seg = segment_table(joined, "trailing_eps_bucket")
    fe_seg = segment_table(joined, "forward_eps_bucket")
    filter_rows = simulate_filters(joined)
    write_summary_md(joined, filter_rows, pe_seg, te_seg, fe_seg, REPORT_MD)

    print(f"Wrote {REPORT_CSV}")
    print(f"Wrote {REPORT_MD}")
    improves = [r for r in filter_rows if r.get("verdict") == "improves"]
    print(f"Improve hits: {len(improves)}")
    for r in improves:
        print(
            f"  {r['filter']} {r['tf']}: WR={r['winRatePct']} "
            f"ΔWR={r['deltaWinRatePct']} P&L={r['pnlUsd']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
