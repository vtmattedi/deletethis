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
    "500-1k_std": "band_500_1k_std",
    "1k-2k_std": "band_1k_2k_std",
    "spectral_flux": "spectral_flux",
    "spectral_flatness": "spectral_flatness",
}

HISTORY_ADDITIONS = {
    "band_500_1k_std": "REAL",
    "band_1k_2k_std": "REAL",
    "spectral_flux": "REAL",
    "spectral_flatness": "REAL",
}

# stream_connected is gone: rows are only written while audio is
# arriving, so it was always 1 and downtime showed up as a hole in the
# timestamps rather than as rows saying so. Outages are recorded
# properly in the `connection` table instead.
COLUMNS = [
    "t",
    "state",
    "candidate",
    "stable_seconds",
    "rms",
    *BAND_COLUMNS.values(),
    "lost_frames",
    "device_dropped",
]

# Averaged when a bucket covers several rows.
NUMERIC_COLUMNS = [
    "stable_seconds",
    "rms",
    *BAND_COLUMNS.values(),
]

# Cumulative counters: the largest in the bucket is the one that counts.
COUNTER_COLUMNS = ["lost_frames", "device_dropped"]

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
    band_500_1k_std  REAL,
    band_1k_2k_std   REAL,
    spectral_flux    REAL,
    spectral_flatness REAL,
    stream_connected INTEGER,
    lost_frames      INTEGER,
    device_dropped   INTEGER
);

CREATE TABLE IF NOT EXISTS connection (
    t         REAL PRIMARY KEY,
    connected INTEGER,
    detail    TEXT
);
"""

DEFAULT_INTERVAL_SECONDS = 1.0

# A plot cannot use more points than it has pixels. Beyond this the
# range is bucketed rather than cut short: a day must look like a day,
# not like the first few hours of one.
MAX_POINTS = 2000


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
            existing = {
                row[1] for row in self.connection.execute(
                    "PRAGMA table_info(history)"
                )
            }
            for column, sql_type in HISTORY_ADDITIONS.items():
                if column not in existing:
                    self.connection.execute(
                        f"ALTER TABLE history ADD COLUMN {column} {sql_type}"
                    )
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
                round(
                    features.get(
                        band,
                        -120.0 if band in {
                            "30-80", "500-1k", "1k-2k", "200-1200"
                        } else 0.0,
                    ),
                    4,
                )
                for band in BAND_COLUMNS
            ],
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
        max_points: int = MAX_POINTS,
    ) -> dict:
        """Columnar, because the only consumer is a chart.

        Over a long range the rows are bucketed rather than cut off. A
        LIMIT would have returned the first few hours of a day and
        called it a day, which is worse than useless: it looks like
        data.
        """
        max_points = max(2, min(max_points, MAX_POINTS))

        with self.lock:
            total = self.connection.execute(
                "SELECT COUNT(*) FROM history WHERE t >= ? AND t <= ?",
                (start, end),
            ).fetchone()[0]

        if total <= max_points:
            rows = self._raw(start, end)
            bucket = 0.0
        else:
            bucket = (end - start) / max_points
            rows = self._bucketed(start, end, bucket)

        columns: dict[str, list] = {name: [] for name in COLUMNS}

        for row in rows:
            for name in COLUMNS:
                value = row[name]

                columns[name].append(
                    round(value, 2)
                    if isinstance(value, float) and name != "t"
                    else value
                )

        return {
            "from": start,
            "to": end,
            "count": len(rows),
            "total": total,
            "downsampled": bucket > 0.0,
            "bucketSeconds": round(bucket, 3),
            "columns": columns,
            "connection": self.connection_events(start, end),
        }

    def _raw(self, start: float, end: float) -> list:
        names = ", ".join(COLUMNS)

        with self.lock:
            return self.connection.execute(
                f"SELECT {names} FROM history "
                "WHERE t >= ? AND t <= ? ORDER BY t",
                (start, end),
            ).fetchall()

    def _bucketed(
        self, start: float, end: float, bucket: float
    ) -> list:
        """One row per bucket: averages, plus the bucket's last state.

        The bare `state` and `candidate` columns take their values from
        the row holding MAX(t), which is SQLite's documented behaviour
        when a query has exactly one max() aggregate. A state change
        inside a bucket is therefore rounded to the end of it -- fine
        for an overview, and the event list is where exact transitions
        live anyway.
        """
        averages = ", ".join(
            f"AVG({name}) AS {name}" for name in NUMERIC_COLUMNS
        )

        counters = ", ".join(
            f"MAX({name}) AS {name}" for name in COUNTER_COLUMNS
        )

        with self.lock:
            return self.connection.execute(
                f"SELECT MAX(t) AS t, state, candidate, "
                f"{averages}, {counters} "
                "FROM history WHERE t >= ? AND t <= ? "
                "GROUP BY CAST((t - ?) / ? AS INTEGER) "
                "ORDER BY t",
                (start, end, start, bucket),
            ).fetchall()

    # ------------------------------------------ connection events

    def record_connection(
        self, connected: bool, detail: str = ""
    ) -> None:
        """Note that the stream came up or went down.

        History rows are only written while audio is arriving, so an
        outage is a hole in the timestamps. These rows say what the
        hole was, which a hole cannot.
        """
        moment = time.time()

        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO connection "
                "(t, connected, detail) VALUES (?, ?, ?)",
                (moment, 1 if connected else 0, detail),
            )
            self.connection.commit()

    def connection_events(
        self, start: float, end: float, limit: int = 500
    ) -> list[dict]:
        """Events in the range, plus the one before it.

        Without the preceding event the browser cannot know whether the
        range opened connected or disconnected.
        """
        with self.lock:
            before = self.connection.execute(
                "SELECT t, connected, detail FROM connection "
                "WHERE t < ? ORDER BY t DESC LIMIT 1",
                (start,),
            ).fetchall()

            inside = self.connection.execute(
                "SELECT t, connected, detail FROM connection "
                "WHERE t >= ? AND t <= ? ORDER BY t LIMIT ?",
                (start, end, limit),
            ).fetchall()

        return [
            {
                "t": row["t"],
                "connected": bool(row["connected"]),
                "detail": row["detail"] or "",
            }
            for row in list(before) + list(inside)
        ]

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
