"""
Ticker source abstraction: strategy-style sources return (tickers, error_message).
No UI or Streamlit; callers show errors.
"""
from typing import Callable, Protocol


class TickerSource(Protocol):
    """Returns (tickers, error_message). error_message is None on success."""

    def get_tickers(self) -> tuple[list[str], str | None]:
        ...

    def description(self) -> str:
        ...


class FileListSource:
    def __init__(self, filename: str, read_list_file_fn: Callable[[str], tuple[list[str], str | None]]):
        self.filename = filename
        self._read = read_list_file_fn

    def get_tickers(self) -> tuple[list[str], str | None]:
        return self._read(self.filename)

    def description(self) -> str:
        return f"Uses {self.filename}. Edit file — next START uses new tickers."


class MergedListSource:
    def __init__(self, filenames: list[str], read_list_file_fn: Callable[[str], tuple[list[str], str | None]]):
        self.filenames = filenames
        self._read = read_list_file_fn

    def get_tickers(self) -> tuple[list[str], str | None]:
        out = []
        first_error = None
        for fn in self.filenames:
            tickers, err = self._read(fn)
            if err and first_error is None:
                first_error = err
            out.extend(tickers)
        # deduplicate preserve order
        return list(dict.fromkeys(out)), first_error

    def description(self) -> str:
        return "Uses SMALL CAP + BIG CAP + ETFS lists merged (no duplicates)."


class ManualSource:
    def __init__(self, get_text_fn: Callable[[], str]):
        self._get_text = get_text_fn

    def get_tickers(self) -> tuple[list[str], str | None]:
        text = self._get_text()
        tickers = [x.strip().upper() for x in text.split(",") if x.strip()]
        return tickers, None

    def description(self) -> str:
        return "Comma-separated symbols. Next START scans these tickers."
