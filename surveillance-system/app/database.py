# import sqlite3
# import datetime
# import json
# from config import DB_PATH


# def _connect():
#     conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#     conn.row_factory = sqlite3.Row
#     return conn


# def init_db():
#     conn = _connect()
#     cur = conn.cursor()
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS events (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             camera TEXT,
#             person TEXT,
#             event TEXT NOT NULL,
#             severity TEXT,
#             confidence REAL,
#             explanation TEXT,
#             snapshot_path TEXT,
#             clip_path TEXT,
#             metadata TEXT
#         )
#     """)
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS alerts (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             event_id INTEGER,
#             channel TEXT,
#             status TEXT,
#             sent_at TEXT,
#             FOREIGN KEY(event_id) REFERENCES events(id)
#         )
#     """)
#     conn.commit()
#     conn.close()


# def log_event(person, event, severity="medium", confidence=0.0,
#               explanation="", snapshot_path=None, clip_path=None,
#               camera="Camera 1", metadata=None):
#     conn = _connect()
#     cur = conn.cursor()
#     cur.execute("""
#         INSERT INTO events (timestamp, camera, person, event, severity,
#                             confidence, explanation, snapshot_path, clip_path, metadata)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#     """, (
#         datetime.datetime.now().isoformat(),
#         camera, person, event, severity, confidence,
#         explanation, snapshot_path, clip_path,
#         json.dumps(metadata or {})
#     ))
#     event_id = cur.lastrowid
#     conn.commit()
#     conn.close()
#     return event_id


# def log_alert(event_id, channel, status):
#     conn = _connect()
#     conn.execute(
#         "INSERT INTO alerts (event_id, channel, status, sent_at) VALUES (?, ?, ?, ?)",
#         (event_id, channel, status, datetime.datetime.now().isoformat())
#     )
#     conn.commit()
#     conn.close()


# def get_events(limit=100, event_type=None, severity=None, person=None):
#     conn = _connect()
#     query = "SELECT * FROM events WHERE 1=1"
#     params = []
#     if event_type:
#         query += " AND event = ?"
#         params.append(event_type)
#     if severity:
#         query += " AND severity = ?"
#         params.append(severity)
#     if person:
#         query += " AND person = ?"
#         params.append(person)
#     query += " ORDER BY id DESC LIMIT ?"
#     params.append(limit)
#     rows = conn.execute(query, params).fetchall()
#     conn.close()
#     return [dict(r) for r in rows]


# def get_event(event_id):
#     conn = _connect()
#     row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
#     conn.close()
#     return dict(row) if row else None


# def get_stats():
#     conn = _connect()
#     total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
#     by_type = conn.execute(
#         "SELECT event, COUNT(*) as c FROM events GROUP BY event"
#     ).fetchall()
#     by_severity = conn.execute(
#         "SELECT severity, COUNT(*) as c FROM events GROUP BY severity"
#     ).fetchall()
#     conn.close()
#     return {
#         "total": total,
#         "by_type": {r["event"]: r["c"] for r in by_type},
#         "by_severity": {r["severity"]: r["c"] for r in by_severity},
#     }



import sqlite3
import datetime
import json
import numpy as np

from config import DB_PATH


def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


class _SafeEncoder(json.JSONEncoder):
    """Handles numpy ints/floats/arrays and other non-JSON types."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def _safe_json(obj):
    """Serialize obj to JSON, tolerating numpy types and cycles."""
    try:
        return json.dumps(obj, cls=_SafeEncoder, default=str)
    except (ValueError, TypeError):
        # Last-resort fallback if something truly circular remains
        return json.dumps({"error": "metadata not serializable"}, default=str)


def init_db():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera TEXT,
            person TEXT,
            event TEXT NOT NULL,
            severity TEXT,
            confidence REAL,
            explanation TEXT,
            snapshot_path TEXT,
            clip_path TEXT,
            metadata TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            channel TEXT,
            status TEXT,
            sent_at TEXT,
            FOREIGN KEY(event_id) REFERENCES events(id)
        )
    """)
    conn.commit()
    conn.close()


def log_event(person, event, severity="medium", confidence=0.0,
              explanation="", snapshot_path=None, clip_path=None,
              camera="Camera 1", metadata=None):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (timestamp, camera, person, event, severity,
                            confidence, explanation, snapshot_path, clip_path, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        camera, person, event, severity, float(confidence) if confidence else 0.0,
        explanation, snapshot_path, clip_path,
        _safe_json(metadata or {})
    ))
    event_id = cur.lastrowid
    conn.commit()
    conn.close()
    return event_id


def log_alert(event_id, channel, status):
    conn = _connect()
    conn.execute(
        "INSERT INTO alerts (event_id, channel, status, sent_at) VALUES (?, ?, ?, ?)",
        (event_id, channel, status, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_events(limit=100, event_type=None, severity=None, person=None):
    conn = _connect()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if event_type:
        query += " AND event = ?"
        params.append(event_type)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if person:
        query += " AND person = ?"
        params.append(person)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_event(event_id):
    conn = _connect()
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats():
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    by_type = conn.execute(
        "SELECT event, COUNT(*) as c FROM events GROUP BY event"
    ).fetchall()
    by_severity = conn.execute(
        "SELECT severity, COUNT(*) as c FROM events GROUP BY severity"
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "by_type": {r["event"]: r["c"] for r in by_type},
        "by_severity": {r["severity"]: r["c"] for r in by_severity},
    }