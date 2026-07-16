"""
Yahoo OHLC validation with rate-limit friendly pacing.
Shared by stock-tickers merge and full US+TSX universe builders.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import yfinance as yf

from data_utils import fill_last_bar_ohlc, resample_to_timeframe, split_batch_ohlcv

# Yahoo rate-limit friendly — soft-fails (no exception) are common
CHUNK_SIZE = 10
BATCH_PAUSE_SEC = 5.0
RATE_LIMIT_COOLDOWN_SEC = 60.0
RATE_LIMIT_BACKOFF_SEC = (20.0, 45.0, 90.0)
MISS_RATE_LIMIT_FRAC = 0.4
MIN_BARS = 50
FETCH_PERIOD = "10y"
INTERVAL = "1d"
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
TIMEFRAMES = ("Daily", "Weekly", "Monthly")
RETRYABLE_REASONS = frozenset({"NO_DAILY", "RATE_LIMIT"})

# (tv_symbol, yahoo, company_name)
TickerEntry = tuple[str, str, str]


def is_rate_limit(exc: BaseException | None = None, msg: str = "") -> bool:
    text = f"{type(exc).__name__ if exc else ''} {exc or ''} {msg}"
    return "RateLimit" in text or "Too Many Requests" in text or "rate limited" in text.lower()


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"checked": {}, "passed": []}
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"checked": {}, "passed": []}


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def ohlc_ok_all_timeframes(df_daily) -> tuple[bool, str]:
    """Match scanner OHLC requirements for Daily / Weekly / Monthly."""
    if df_daily is None or getattr(df_daily, "empty", True):
        return False, "NO_DAILY"

    for tf in TIMEFRAMES:
        df = df_daily.copy()
        if tf != "Daily":
            df = resample_to_timeframe(df, tf)
        if df is None or df.empty:
            return False, f"NO_OHLC_{tf.upper()}"
        df = fill_last_bar_ohlc(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < MIN_BARS:
            return False, f"INSUFFICIENT_{tf.upper()}"

    return True, "PASS"


def _yf_download(batch: list[str]):
    if len(batch) == 1:
        return yf.download(
            batch[0],
            period=FETCH_PERIOD,
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
            threads=False,
        )
    return yf.download(
        batch,
        period=FETCH_PERIOD,
        interval=INTERVAL,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=False,
    )


def fetch_batch_dfs(batch: list[str]) -> tuple[dict, bool]:
    """Download batch with retries. Returns (yahoo -> DataFrame, rate_limited)."""
    ticker_dfs: dict = {}
    rate_limited = False

    for attempt in range(len(RATE_LIMIT_BACKOFF_SEC) + 1):
        try:
            data = _yf_download(batch)
        except Exception as exc:
            if is_rate_limit(exc):
                rate_limited = True
                if attempt < len(RATE_LIMIT_BACKOFF_SEC):
                    delay = RATE_LIMIT_BACKOFF_SEC[attempt]
                    print(f"  Yahoo rate limit (exception) - sleep {delay:.0f}s")
                    time.sleep(delay)
                    continue
                return {}, True
            if attempt < len(RATE_LIMIT_BACKOFF_SEC):
                time.sleep(3.0 * (attempt + 1))
                continue
            return {}, False

        if data is None or getattr(data, "empty", True):
            rate_limited = True
            if attempt < len(RATE_LIMIT_BACKOFF_SEC):
                delay = RATE_LIMIT_BACKOFF_SEC[attempt]
                print(f"  Empty Yahoo batch - sleep {delay:.0f}s (likely rate limit)")
                time.sleep(delay)
                continue
            return {}, True

        if len(batch) == 1:
            yahoo = batch[0]
            if all(col in data.columns for col in REQUIRED_COLS):
                ticker_dfs = {yahoo: data[REQUIRED_COLS].copy()}
            else:
                ticker_dfs = {}
        else:
            ticker_dfs = split_batch_ohlcv(data, batch, REQUIRED_COLS)

        miss = [
            y for y in batch
            if y not in ticker_dfs or getattr(ticker_dfs.get(y), "empty", True)
        ]
        miss_frac = len(miss) / max(len(batch), 1)
        if miss and miss_frac >= MISS_RATE_LIMIT_FRAC:
            rate_limited = True
            if attempt < len(RATE_LIMIT_BACKOFF_SEC):
                delay = RATE_LIMIT_BACKOFF_SEC[attempt]
                print(
                    f"  High miss rate {len(miss)}/{len(batch)} - "
                    f"sleep {delay:.0f}s then retry batch"
                )
                time.sleep(delay)
                continue
            return ticker_dfs, True

        return ticker_dfs, False

    return ticker_dfs, rate_limited


def invalidate_retryable(checked: dict) -> int:
    drop = [k for k, v in checked.items() if v.get("reason") in RETRYABLE_REASONS]
    for k in drop:
        del checked[k]
    return len(drop)


def validate_ohlc_candidates(
    candidates: list[TickerEntry],
    *,
    cache_path: Path,
    resume: bool = True,
    retry_no_data: bool = False,
    progress_every: int = 25,
) -> tuple[list[TickerEntry], Counter]:
    cache = load_cache(cache_path) if resume else {"checked": {}, "passed": []}
    checked: dict = cache.setdefault("checked", {})
    if retry_no_data:
        n = invalidate_retryable(checked)
        print(f"Invalidated {n} retryable cache entries (NO_DAILY / RATE_LIMIT)")

    passed: list[list[str]] = []
    passed_yahoo: set[str] = set()
    rejects: Counter = Counter()

    pending: list[TickerEntry] = []
    for tv_sym, yahoo, name_hint in candidates:
        if yahoo in checked:
            prev = checked[yahoo]
            reason = prev.get("reason", "NO_DAILY")
            if reason == "PASS":
                if yahoo not in passed_yahoo:
                    passed.append(
                        [
                            prev.get("tv_part", tv_sym),
                            yahoo,
                            prev.get("name", name_hint),
                        ]
                    )
                    passed_yahoo.add(yahoo)
            else:
                rejects[reason] += 1
        else:
            pending.append((tv_sym, yahoo, name_hint))

    print(f"OHLC check pending: {len(pending)} (cached: {len(candidates) - len(pending)})")
    print(
        f"Yahoo pacing: chunk={CHUNK_SIZE}, batch_pause={BATCH_PAUSE_SEC}s, "
        f"threads=False, no single-ticker hammering"
    )

    done = len(candidates) - len(pending)
    consecutive_rate_limits = 0

    for batch_start in range(0, len(pending), CHUNK_SIZE):
        if batch_start > 0:
            time.sleep(BATCH_PAUSE_SEC)

        batch_entries = pending[batch_start : batch_start + CHUNK_SIZE]
        batch = [yahoo for _, yahoo, _ in batch_entries]
        ticker_dfs, rate_limited = fetch_batch_dfs(batch)

        if rate_limited:
            consecutive_rate_limits += 1
            if consecutive_rate_limits >= 2:
                print(f"  Cooling down {RATE_LIMIT_COOLDOWN_SEC:.0f}s after repeated rate limits...")
                time.sleep(RATE_LIMIT_COOLDOWN_SEC)
                consecutive_rate_limits = 0
                ticker_dfs, rate_limited = fetch_batch_dfs(batch)
        else:
            consecutive_rate_limits = 0

        for tv_sym, yahoo, name_hint in batch_entries:
            df = ticker_dfs.get(yahoo)
            if df is None or getattr(df, "empty", True):
                reason = "RATE_LIMIT" if rate_limited else "NO_DAILY"
                ok = False
            else:
                ok, reason = ohlc_ok_all_timeframes(df)

            checked[yahoo] = {
                "reason": reason,
                "tv_part": tv_sym,
                "name": name_hint,
            }
            if ok:
                passed.append([tv_sym, yahoo, name_hint])
                passed_yahoo.add(yahoo)
            else:
                rejects[reason] += 1

            done += 1
            if progress_every > 0 and done % progress_every == 0:
                save_cache(cache_path, {"checked": checked, "passed": passed})
                print(f"  OHLC checked {done}/{len(candidates)} - passed so far: {len(passed)}")

        save_cache(cache_path, {"checked": checked, "passed": passed})

    entries: list[TickerEntry] = [
        (tv_part, yahoo, name) for tv_part, yahoo, name in passed
    ]
    entries.sort(key=lambda e: e[0])
    return entries, rejects
