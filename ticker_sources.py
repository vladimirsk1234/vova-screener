"""
Ticker source abstraction: strategy-style sources return (tickers, tv_map, name_map, error_message).
No UI or Streamlit; callers show errors.
"""
from __future__ import annotations

from typing import Callable, Protocol


class TickerSource(Protocol):
    """Returns (tickers, tv_symbol_by_yahoo, company_name_by_yahoo, error_message). error_message is None on success."""

    def get_tickers(self) -> tuple[list[str], dict[str, str], dict[str, str], str | None]:
        ...

    def description(self) -> str:
        ...


class FileListSource:
    def __init__(
        self,
        filename: str,
        read_list_file_fn: Callable[[str], tuple[list[str], dict[str, str], dict[str, str], str | None]],
    ):
        self.filename = filename
        self._read = read_list_file_fn

    def get_tickers(self) -> tuple[list[str], dict[str, str], dict[str, str], str | None]:
        return self._read(self.filename)

    def description(self) -> str:
        return f"Uses {self.filename}. Edit file — next START uses new tickers."


class ManualSource:
    def __init__(self, get_text_fn: Callable[[], str]):
        self._get_text = get_text_fn

    def get_tickers(self) -> tuple[list[str], dict[str, str], dict[str, str], str | None]:
        text = self._get_text()
        tickers = [x.strip().upper() for x in text.split(",") if x.strip()]
        return tickers, {}, {}, None

    def description(self) -> str:
        return "Comma-separated symbols. Next START scans these tickers."


class CombinedListSource:
    """Merge multiple ticker sources (dedupe by Yahoo symbol, first wins)."""

    def __init__(self, sources: list[TickerSource], label: str = "combined list"):
        self._sources = sources
        self._label = label

    def get_tickers(self) -> tuple[list[str], dict[str, str], dict[str, str], str | None]:
        tickers: list[str] = []
        tv_map: dict[str, str] = {}
        name_map: dict[str, str] = {}
        seen: set[str] = set()
        for src in self._sources:
            t_list, tv, names, err = src.get_tickers()
            if err:
                return [], {}, {}, err
            for t in t_list:
                if t in seen:
                    continue
                seen.add(t)
                tickers.append(t)
                if t in tv:
                    tv_map[t] = tv[t]
                if t in names:
                    name_map[t] = names[t]
        return tickers, tv_map, name_map, None

    def description(self) -> str:
        return f"Combined {self._label}. Edit list files — next START uses new tickers."
