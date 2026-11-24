"""What the proxy did, request by request, and the keys it accepts: one SQLite file next to its state.

Nothing a customer sent is stored here, only what happened to it: who, which decision, who answered (vega, the
LLM, or nobody because it passed straight through), the model, how long it took, the tokens and dollars, what
vega saved, its confidence and why it escalated. That is enough for usage charts, per-user tables and request
logs, and leaves the messages themselves where they were.

    led = Ledger("./vega-proxy/ledger.db")
    key, row = led.create_key("user_2abc", "production")   # the key is shown once; only its hash is kept
    led.tenant_for(key)                                    # -> "user_2abc"
    led.log(tenant="user_2abc", decision="support-route", source="vega", latency_ms=41.0, saved_usd=0.008)
    led.usage(owner="user_2abc", days=30)                  # totals, daily series, per decision, per model
    led.set_limit("user_2abc", 10_000)                     # at most 10,000 vega answers a calendar month
    led.over_limit("user_2abc")                            # -> True once they are used up
"""

import calendar
import hashlib
import secrets
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    id TEXT PRIMARY KEY, hash TEXT UNIQUE NOT NULL, prefix TEXT NOT NULL, owner TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '', created REAL NOT NULL, revoked REAL, last_used REAL
);
CREATE INDEX IF NOT EXISTS keys_owner ON keys(owner);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, tenant TEXT NOT NULL, key_id TEXT,
    decision TEXT, api TEXT, model TEXT, source TEXT NOT NULL, mode TEXT, status INTEGER,
    latency_ms REAL, in_tokens INTEGER, out_tokens INTEGER, cost_usd REAL, saved_usd REAL,
    confidence REAL, escalated TEXT, read TEXT
);
CREATE INDEX IF NOT EXISTS events_tenant_ts ON events(tenant, ts);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS limits (
    owner TEXT PRIMARY KEY, vega_limit INTEGER NOT NULL, since REAL, updated REAL NOT NULL
);
"""
_FIELDS = (
    "ts",
    "tenant",
    "key_id",
    "decision",
    "api",
    "model",
    "source",
    "mode",
    "status",
    "latency_ms",
    "in_tokens",
    "out_tokens",
    "cost_usd",
    "saved_usd",
    "confidence",
    "escalated",
    "read",
)


def _hash(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def month_start(now=None):
    """The first second of the current calendar month, UTC."""
    t = time.gmtime(time.time() if now is None else now)
    return float(calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0)))


def next_month_start(now=None):
    t = time.gmtime(time.time() if now is None else now)
    y, m = (t.tm_year + 1, 1) if t.tm_mon == 12 else (t.tm_year, t.tm_mon + 1)
    return float(calendar.timegm((y, m, 1, 0, 0, 0)))


class Ledger:
    def __init__(self, path):
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
        self._cache = {}  # key hash -> (key id, owner), revocation clears it
        self._limits = None  # owner -> (vega_limit, since or None), read once
        self._used = {}  # owner -> [window start, vega answers in it], kept in step by log()

    # ------------------------------------------------------------------ keys
    def create_key(self, owner, name=""):
        """A new key for `owner`. Returns (the key, its row); the key itself is never stored and cannot be
        shown again."""
        key = "vg_" + secrets.token_hex(20)
        row = {
            "id": "key_" + secrets.token_hex(8),
            "prefix": key[:10],
            "owner": str(owner),
            "name": str(name or "")[:80],
            "created": time.time(),
            "revoked": None,
            "last_used": None,
        }
        with self._lock:
            self._db.execute(
                "INSERT INTO keys (id, hash, prefix, owner, name, created) VALUES (?, ?, ?, ?, ?, ?)",
                (row["id"], _hash(key), row["prefix"], row["owner"], row["name"], row["created"]),
            )
        return key, row

    def list_keys(self, owner=None):
        q = "SELECT id, prefix, owner, name, created, revoked, last_used FROM keys"
        args = ()
        if owner is not None:
            q, args = q + " WHERE owner = ?", (owner,)
        with self._lock:
            return [dict(r) for r in self._db.execute(q + " ORDER BY created DESC", args)]

    def revoke_key(self, key_id, owner=None):
        """True when a live key was revoked. With `owner`, only that owner's key can be."""
        q, args = "UPDATE keys SET revoked = ? WHERE id = ? AND revoked IS NULL", [time.time(), key_id]
        if owner is not None:
            q, args = q + " AND owner = ?", args + [owner]
        with self._lock:
            n = self._db.execute(q, args).rowcount
        self._cache.clear()
        return n > 0

    def has_keys(self):
        with self._lock:
            # any key ever made, revoked ones too: a proxy that has handed out keys never falls back to open
            return self._db.execute("SELECT 1 FROM keys LIMIT 1").fetchone() is not None

    def tenant_for(self, key):
        """(key id, owner) for a live key, or None."""
        h = _hash(key or "")
        got = self._cache.get(h)
        if got is None:
            with self._lock:
                r = self._db.execute(
                    "SELECT id, owner FROM keys WHERE hash = ? AND revoked IS NULL", (h,)
                ).fetchone()
            if r is None:
                return None
            got = self._cache[h] = (r["id"], r["owner"])
            with self._lock:
                self._db.execute("UPDATE keys SET last_used = ? WHERE id = ?", (time.time(), r["id"]))
        return got

    # ------------------------------------------------------------------ limits
    def set_limit(self, owner, vega_limit, since=None):
        """At most `vega_limit` answers by vega for `owner` per period. Past that the proxy sends every
        decision to the LLM: nothing is refused, vega just stops answering. The period is the calendar month
        (UTC), or, with `since` (a timestamp), everything from then on. `vega_limit=None` removes it."""
        owner = str(owner)
        with self._lock:
            if vega_limit is None:
                self._db.execute("DELETE FROM limits WHERE owner = ?", (owner,))
            else:
                self._db.execute(
                    "INSERT INTO limits (owner, vega_limit, since, updated) VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(owner) DO UPDATE SET vega_limit = excluded.vega_limit,"
                    " since = excluded.since, updated = excluded.updated",
                    (owner, max(0, int(vega_limit)), None if since is None else float(since), time.time()),
                )
            self._limits = None
            self._used.pop(owner, None)

    def _limit_rows(self):
        if self._limits is None:
            with self._lock:
                rows = self._db.execute("SELECT owner, vega_limit, since FROM limits").fetchall()
            self._limits = {r["owner"]: (int(r["vega_limit"]), r["since"]) for r in rows}
        return self._limits

    def vega_count(self, owner, since, until=None):
        q, args = (
            "SELECT COUNT(*) FROM events WHERE tenant = ? AND source = 'vega' AND ts >= ?",
            [owner, since],
        )
        if until is not None:
            q, args = q + " AND ts < ?", args + [until]
        with self._lock:
            return int(self._db.execute(q, args).fetchone()[0])

    def _window_used(self, owner, since):
        """vega answers for `owner` since `since`: counted once from the events, then kept in step by
        log()."""
        got = self._used.get(owner)
        if got is None or got[0] != since:
            got = self._used[owner] = [since, self.vega_count(owner, since)]
        return got[1]

    def get_limit(self, owner):
        """None when `owner` has no limit, else {"vega_limit", "used", "left", "reached", "since", "resets"}
        (`resets` is None for a limit counted from a fixed `since`)."""
        row = self._limit_rows().get(str(owner))
        if row is None:
            return None
        cap, fixed = row
        since = month_start() if fixed is None else float(fixed)
        used = self._window_used(str(owner), since)
        return {
            "owner": str(owner),
            "vega_limit": cap,
            "used": used,
            "left": max(0, cap - used),
            "reached": used >= cap,
            "since": since,
            "resets": next_month_start() if fixed is None else None,
        }

    def limits(self):
        return [self.get_limit(o) for o in sorted(self._limit_rows())]

    def over_limit(self, owner):
        """True when `owner` has a limit and has used it up. Cheap: no query after the first call in a
        period."""
        if not self._limit_rows():
            return False
        got = self.get_limit(owner)
        return bool(got and got["reached"])

    # ------------------------------------------------------------------ events
    def log(self, **e):
        e.setdefault("ts", time.time())
        row = tuple(e.get(f) for f in _FIELDS)
        with self._lock:
            self._db.execute(
                "INSERT INTO events (%s) VALUES (%s)" % (", ".join(_FIELDS), ", ".join("?" * len(_FIELDS))),
                row,
            )
            got = self._used.get(e.get("tenant"))
            if got is not None and e.get("source") == "vega" and e["ts"] >= got[0]:
                got[1] += 1

    def _where(self, owner, since, until=None):
        q, args = " WHERE ts >= ?", [since]
        if until is not None:
            q, args = q + " AND ts < ?", args + [until]
        if owner is not None:
            q, args = q + " AND tenant = ?", args + [owner]
        return q, args

    def usage(self, owner=None, days=30, since=None, until=None):
        """Totals, a daily series, and breakdowns by decision and model (and by user when `owner` is None).
        The window is the last `days`, or exactly [`since`, `until`) when `since` (a timestamp) is given: what
        a billing period needs."""
        if since is None:
            since, until = time.time() - max(1, int(days)) * 86400, None
        else:
            since, until = float(since), (None if until is None else float(until))
        w, args = self._where(owner, since, until)
        agg = (
            "COUNT(*) AS requests,"
            " SUM(source = 'vega') AS vega, SUM(source = 'llm') AS llm,"
            " SUM(source = 'passthrough') AS passthrough,"
            " SUM(source = 'error' OR status >= 400) AS errors,"
            " SUM(COALESCE(cost_usd, 0)) AS cost_usd, SUM(COALESCE(saved_usd, 0)) AS saved_usd,"
            " AVG(CASE WHEN source = 'vega' THEN latency_ms END) AS vega_ms,"
            " AVG(CASE WHEN source = 'llm' THEN latency_ms END) AS llm_ms,"
            " AVG(CASE WHEN source IN ('vega', 'llm') THEN confidence END) AS confidence"
        )
        with self._lock:
            totals = dict(self._db.execute("SELECT " + agg + " FROM events" + w, args).fetchone())
            daily = [
                dict(r)
                for r in self._db.execute(
                    "SELECT date(ts, 'unixepoch') AS day, "
                    + agg
                    + " FROM events"
                    + w
                    + " GROUP BY day ORDER BY day",
                    args,
                )
            ]
            by_decision = [
                dict(r)
                for r in self._db.execute(
                    "SELECT decision, "
                    + agg
                    + " FROM events"
                    + w
                    + " AND decision IS NOT NULL GROUP BY decision"
                    " ORDER BY requests DESC LIMIT 100",
                    args,
                )
            ]
            by_model = [
                dict(r)
                for r in self._db.execute(
                    "SELECT model, "
                    + agg
                    + " FROM events"
                    + w
                    + " GROUP BY model ORDER BY requests DESC LIMIT 50",
                    args,
                )
            ]
            by_user = []
            if owner is None:
                by_user = [
                    dict(r)
                    for r in self._db.execute(
                        "SELECT tenant, MAX(ts) AS last_seen, "
                        + agg
                        + " FROM events"
                        + w
                        + " GROUP BY tenant ORDER BY requests DESC LIMIT 500",
                        args,
                    )
                ]
            reasons = [
                dict(r)
                for r in self._db.execute(
                    "SELECT escalated AS reason, COUNT(*) AS n FROM events"
                    + w
                    + " AND escalated IS NOT NULL AND escalated != ''"
                    + " GROUP BY escalated ORDER BY n DESC LIMIT 20",
                    args,
                )
            ]
        for d in [totals] + daily + by_decision + by_model + by_user:
            for k in ("vega", "llm", "passthrough", "errors"):
                d[k] = int(d.get(k) or 0)
            decided = d["vega"] + d["llm"]
            d["vega_share"] = round(d["vega"] / decided, 4) if decided else 0.0
            for k in ("cost_usd", "saved_usd"):
                d[k] = round(float(d.get(k) or 0.0), 6)
            for k in ("vega_ms", "llm_ms", "confidence"):
                d[k] = None if d.get(k) is None else round(float(d[k]), 3)
        return {
            "days": int(days),
            "since": since,
            "until": until,
            "owner": owner,
            "totals": totals,
            "daily": daily,
            "decisions": by_decision,
            "models": by_model,
            "users": by_user,
            "escalation_reasons": reasons,
        }

    def logs(self, owner=None, limit=100, before=None, decision=None, source=None):
        """The newest events, newest first; `before` (an event id) pages back."""
        q, args = "SELECT * FROM events WHERE 1 = 1", []
        if owner is not None:
            q, args = q + " AND tenant = ?", args + [owner]
        if before is not None:
            q, args = q + " AND id < ?", args + [int(before)]
        if decision:
            q, args = q + " AND decision = ?", args + [decision]
        if source:
            q, args = q + " AND source = ?", args + [source]
        q += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self._lock:
            return [dict(r) for r in self._db.execute(q, args)]

    def close(self):
        with self._lock:
            self._db.close()
