"""Compact state history in SQLite.

One row per second: enough to plot a day and see when the air
conditioner ran, small enough to ignore. About 86k rows a day, which
SQLite does not notice.

Deliberately not stored: the ~31 classifier windows per second, and
raw PCM. The windows are only interesting around a transition, and
that is what the event WAVs are for.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Band name as the classifier knows it -> column name.
BAND_COLUMNS = {
    "30-80": "band_30_80",
    "500-1k": "band_500_1k",
    "1k-2k": "band_1k_2k",
    "200-1200": "band_200_1200",
}

COLUMNS = [
    "t",
    "state",
    "candidate",
    "stable_seconds",
    "rms",
    *BAND_COLUMNS.values(),
    "stream_connected",
    "lost_frames",
    "device_dropped",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    t                REAL PRIMARY KEY,
    state            TEXT,
    candidate        TEXT,
    stable_seconds   REAL,
    rms              REAL,
    band_30_80       REAL,
    band_500_1k      REAL,
    band_1k_2k       REAL,
    band_200_1200    REAL,
    stream_connected INTEGER,
    lost_frames      INTEGER,
    device_dropped   INTEGER
);
"""

DEFAULT_INTERVAL_SECONDS = 1.0

# A plot cannot use more points than it has pixels, and a wide range
# should not turn into a 100 MB response.
MAX_ROWS = 20000


def to_epoch(value: str | float | None, fallback: float) -> float:
    """Accept a unix timestamp or an ISO-8601 string."""
    if value is None or value == "":
        return fallback

    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    text = str(value).replace("Z", "+00:00")

    moment = datetime.fromisoformat(text)

    if moment.tzinfo is None:
        moment = moment.astimezone()

    return moment.timestamp()


class HistoryStore:
    """Writes from the stream thread, reads from request handlers.

    One connection guarded by a lock. The write rate is one row per
    second, so there is nothing here worth a connection pool.
    """

    def __init__(
        self,
        path: Path,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self.path = path
        self.interval = interval_seconds

        path.parent.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.connection = sqlite3.connect(
            str(path), check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row

        with self.lock:
            # WAL so a long read cannot block the once-a-second write.
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.executescript(SCHEMA)
            self.connection.commit()

        self.last_written = 0.0
        self.rows_written = 0

    # ------------------------------------------------------- write

    def maybe_record(
        self,
        snapshot,
        health,
        now: float | None = None,
    ) -> bool:
        """Record a row if the interval has elapsed. Cheap to call."""
        moment = time.time() if now is None else now

        if moment - self.last_written < self.interval:
            return False

        self.last_written = moment

        features = snapshot.features

        values = [
            moment,
            snapshot.state,
            snapshot.candidate,
            round(snapshot.stable_seconds, 2),
            round(features.get("rms", -120.0), 2),
            *[
                round(features.get(band, -120.0), 2)
                for band in BAND_COLUMNS
            ],
            1 if health.connected else 0,
            health.lost_frames,
            health.device_dropped,
        ]

        placeholders = ", ".join("?" * len(COLUMNS))

        with self.lock:
            self.connection.execute(
                f"INSERT OR REPLACE INTO history "
                f"({', '.join(COLUMNS)}) VALUES ({placeholders})",
                values,
            )
            self.connection.commit()

        self.rows_written += 1

        return True

    # -------------------------------------------------------- read

    def query(
        self,
        start: float,
        end: float,
        limit: int = MAX_ROWS,
    ) -> dict:
        """Columnar, because the only consumer is a chart.

        Row-per-object JSON would trade a few times the bytes for a
        shape the browser would immediately have to transpose anyway.
        """
        limit = max(1, min(limit, MAX_ROWS))

        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM history "
                "WHERE t >= ? AND t <= ? "
                "ORDER BY t LIMIT ?",
                (start, end, limit),
            ).fetchall()

        columns: dict[str, list] = {name: [] for name in COLUMNS}

        for row in rows:
            for name in COLUMNS:
                columns[name].append(row[name])

        return {
            "from": start,
            "to": end,
            "count": len(rows),
            "truncated": len(rows) >= limit,
            "columns": columns,
        }

    def span(self) -> dict:
        with self.lock:
            row = self.connection.execute(
                "SELECT MIN(t) AS first, MAX(t) AS last, "
                "COUNT(*) AS rows FROM history"
            ).fetchone()

        return {
            "first": row["first"],
            "last": row["last"],
            "rows": row["rows"] or 0,
        }

    def status(self) -> dict:
        span = self.span()

        return {
            "database": str(self.path),
            "rows": span["rows"],
            "first": (
                datetime.fromtimestamp(
                    span["first"], timezone.utc
                ).isoformat(timespec="seconds")
                if span["first"]
                else None
            ),
            "intervalSeconds": self.interval,
        }

    def close(self) -> None:
        with self.lock:
            self.connection.close()
