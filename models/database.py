import sqlite3
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = os.environ.get('DATABASE_PATH', 'factoryiq.db')

def get_db_path():
    return os.environ.get('DATABASE_PATH', DB_PATH)

@contextmanager
def get_db():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initialize all database tables."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS machines (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            type        TEXT,
            location    TEXT,
            status      TEXT DEFAULT 'healthy',
            last_maintenance TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sensor_data (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            temperature     REAL,
            pressure        REAL,
            vibration       REAL,
            humidity        REAL,
            speed           REAL,
            production_rate REAL,
            quality_score   REAL,
            is_simulated    INTEGER DEFAULT 1,
            FOREIGN KEY (machine_id) REFERENCES machines(id)
        );

        CREATE TABLE IF NOT EXISTS production_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            batch_id        TEXT,
            quantity        INTEGER,
            defect_count    INTEGER DEFAULT 0,
            defect_rate     REAL DEFAULT 0.0,
            defect_type     TEXT,
            quality_score   REAL,
            shift           TEXT,
            operator        TEXT,
            notes           TEXT,
            is_simulated    INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            parameter       TEXT,
            severity        TEXT,
            detected_value  REAL,
            normal_min      REAL,
            normal_max      REAL,
            message         TEXT,
            ai_recommendation TEXT,
            status          TEXT DEFAULT 'active',
            acknowledged_at TEXT,
            resolved_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id          TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            defect_probability  REAL,
            risk_category       TEXT,
            predicted_defect    TEXT,
            confidence_score    REAL,
            contributing_factors TEXT,
            ai_explanation      TEXT,
            is_simulated        INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            priority        TEXT,
            issue           TEXT,
            evidence        TEXT,
            action          TEXT,
            expected_impact TEXT,
            status          TEXT DEFAULT 'open',
            acknowledged_at TEXT,
            resolved_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            original_name TEXT NOT NULL,
            doc_type    TEXT,
            content     TEXT,
            chunks      TEXT,
            embeddings  TEXT,
            uploaded_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS agent_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            action      TEXT,
            result      TEXT,
            duration_ms INTEGER,
            machine_id  TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key     TEXT PRIMARY KEY,
            value   TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );
        """)
    _ensure_machines()
    _ensure_settings()

def _ensure_machines():
    """Seed machines table if empty."""
    machines = [
        ('A-101', 'CNC Milling Machine A-101', 'CNC Milling', 'Line A', '2024-11-15'),
        ('A-102', 'CNC Milling Machine A-102', 'CNC Milling', 'Line A', '2024-10-20'),
        ('B-201', 'Injection Molding B-201', 'Injection Molding', 'Line B', '2024-12-01'),
        ('B-202', 'Injection Molding B-202', 'Injection Molding', 'Line B', '2024-09-30'),
        ('C-301', 'Assembly Robot C-301', 'Robotic Assembly', 'Line C', '2024-11-05'),
        ('C-302', 'Assembly Robot C-302', 'Robotic Assembly', 'Line C', '2024-10-10'),
        ('D-401', 'Welding Station D-401', 'Welding', 'Line D', '2024-12-10'),
        ('D-402', 'Welding Station D-402', 'Welding', 'Line D', '2024-11-28'),
        ('E-501', 'Quality Inspection E-501', 'Inspection', 'Line E', '2024-12-05'),
        ('E-502', 'Packaging Line E-502', 'Packaging', 'Line E', '2024-11-18'),
    ]
    with get_db() as conn:
        existing = conn.execute("SELECT COUNT(*) as c FROM machines").fetchone()['c']
        if existing == 0:
            conn.executemany(
                "INSERT OR IGNORE INTO machines (id, name, type, location, last_maintenance) VALUES (?,?,?,?,?)",
                machines
            )

def _ensure_settings():
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('simulation_active', '0')")
        conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('demo_anomaly_machine', '')")

# ─── Generic helpers ───────────────────────────────────────────────────────────

def get_all_machines():
    with get_db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM machines ORDER BY id").fetchall()]

def get_machine(machine_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM machines WHERE id=?", (machine_id,)).fetchone()
        return dict(row) if row else None

def update_machine_status(machine_id, status):
    with get_db() as conn:
        conn.execute("UPDATE machines SET status=? WHERE id=?", (status, machine_id))

def insert_sensor_reading(machine_id, data, is_simulated=True):
    ts = data.get('timestamp', datetime.utcnow().isoformat())
    with get_db() as conn:
        conn.execute("""
            INSERT INTO sensor_data
              (machine_id,timestamp,temperature,pressure,vibration,humidity,speed,production_rate,quality_score,is_simulated)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            machine_id, ts,
            data.get('temperature'), data.get('pressure'), data.get('vibration'),
            data.get('humidity'), data.get('speed'), data.get('production_rate'),
            data.get('quality_score'), int(is_simulated)
        ))

def get_latest_sensor(machine_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sensor_data WHERE machine_id=? ORDER BY timestamp DESC LIMIT 1",
            (machine_id,)
        ).fetchone()
        return dict(row) if row else None

def get_sensor_history(machine_id, limit=100):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sensor_data WHERE machine_id=? ORDER BY timestamp DESC LIMIT ?",
            (machine_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

def insert_alert(machine_id, parameter, severity, detected_value,
                 normal_min, normal_max, message, ai_rec=''):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO alerts
              (machine_id,timestamp,parameter,severity,detected_value,normal_min,normal_max,message,ai_recommendation)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (machine_id, datetime.utcnow().isoformat(), parameter, severity,
              detected_value, normal_min, normal_max, message, ai_rec))

def get_alerts(status=None, limit=200):
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE status=? ORDER BY timestamp DESC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

def update_alert_status(alert_id, status):
    ts = datetime.utcnow().isoformat()
    with get_db() as conn:
        if status == 'acknowledged':
            conn.execute("UPDATE alerts SET status=?, acknowledged_at=? WHERE id=?", (status, ts, alert_id))
        elif status == 'resolved':
            conn.execute("UPDATE alerts SET status=?, resolved_at=? WHERE id=?", (status, ts, alert_id))
        else:
            conn.execute("UPDATE alerts SET status=? WHERE id=?", (status, alert_id))

def insert_prediction(machine_id, defect_prob, risk_cat, pred_defect,
                      confidence, factors, explanation='', is_sim=True):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO predictions
              (machine_id,timestamp,defect_probability,risk_category,predicted_defect,confidence_score,contributing_factors,ai_explanation,is_simulated)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            machine_id, datetime.utcnow().isoformat(),
            defect_prob, risk_cat, pred_defect, confidence,
            json.dumps(factors), explanation, int(is_sim)
        ))

def get_latest_predictions(limit=50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d['contributing_factors'] = json.loads(d['contributing_factors'] or '[]')
            except Exception:
                d['contributing_factors'] = []
            result.append(d)
        return result

def get_machine_predictions(machine_id, limit=20):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE machine_id=? ORDER BY timestamp DESC LIMIT ?",
            (machine_id, limit)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d['contributing_factors'] = json.loads(d['contributing_factors'] or '[]')
            except Exception:
                d['contributing_factors'] = []
            result.append(d)
        return result

def insert_recommendation(machine_id, priority, issue, evidence, action, impact):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO recommendations
              (machine_id,timestamp,priority,issue,evidence,action,expected_impact)
            VALUES (?,?,?,?,?,?,?)
        """, (machine_id, datetime.utcnow().isoformat(), priority, issue, evidence, action, impact))

def get_recommendations(status=None, limit=100):
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM recommendations WHERE status=? ORDER BY timestamp DESC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

def update_recommendation_status(rec_id, status):
    ts = datetime.utcnow().isoformat()
    with get_db() as conn:
        if status == 'resolved':
            conn.execute("UPDATE recommendations SET status=?, resolved_at=? WHERE id=?", (status, ts, rec_id))
        elif status == 'acknowledged':
            conn.execute("UPDATE recommendations SET status=?, acknowledged_at=? WHERE id=?", (status, ts, rec_id))
        else:
            conn.execute("UPDATE recommendations SET status=? WHERE id=?", (status, rec_id))

def log_agent(agent_name, action, result='', duration_ms=0, machine_id=None):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO agent_logs (agent_name,timestamp,action,result,duration_ms,machine_id)
            VALUES (?,?,?,?,?,?)
        """, (agent_name, datetime.utcnow().isoformat(), action, str(result)[:500], duration_ms, machine_id))

def get_agent_logs(agent_name=None, limit=50):
    with get_db() as conn:
        if agent_name:
            rows = conn.execute(
                "SELECT * FROM agent_logs WHERE agent_name=? ORDER BY timestamp DESC LIMIT ?",
                (agent_name, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

def get_setting(key, default=''):
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row['value'] if row else default

def set_setting(key, value):
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), datetime.utcnow().isoformat())
        )

def get_production_stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM production_records").fetchone()['c']
        defective = conn.execute(
            "SELECT SUM(defect_count) as s FROM production_records"
        ).fetchone()['s'] or 0
        avg_quality = conn.execute(
            "SELECT AVG(quality_score) as a FROM production_records"
        ).fetchone()['a'] or 90.0
        return {'total': total, 'defective': int(defective), 'avg_quality': round(avg_quality, 1)}
