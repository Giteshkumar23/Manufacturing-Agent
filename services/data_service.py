"""
Synthetic manufacturing data generator.
Produces realistic multi-machine data with normal and anomalous conditions.
"""
import random
import math
import numpy as np
from datetime import datetime, timedelta
from models.database import (
    get_all_machines, insert_sensor_reading, insert_alert,
    insert_prediction, insert_recommendation, get_db
)
from config import Config

RANGES = Config.NORMAL_RANGES

# ── Per-machine "personality" — slight offsets so machines differ ──────────────
MACHINE_PROFILES = {
    'A-101': {'temp_offset': 0,    'vib_offset': 0,    'health': 0.95},
    'A-102': {'temp_offset': 1.5,  'vib_offset': 0.1,  'health': 0.90},
    'B-201': {'temp_offset': -1,   'vib_offset': -0.1, 'health': 0.97},
    'B-202': {'temp_offset': 2,    'vib_offset': 0.2,  'health': 0.85},
    'C-301': {'temp_offset': 0.5,  'vib_offset': 0,    'health': 0.93},
    'C-302': {'temp_offset': -0.5, 'vib_offset': 0.05, 'health': 0.91},
    'D-401': {'temp_offset': 1,    'vib_offset': 0.15, 'health': 0.88},
    'D-402': {'temp_offset': 0,    'vib_offset': 0,    'health': 0.96},
    'E-501': {'temp_offset': -1,   'vib_offset': -0.05,'health': 0.99},
    'E-502': {'temp_offset': 0.5,  'vib_offset': 0.1,  'health': 0.92},
}


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def generate_reading(machine_id: str, anomaly_mode: bool = False, severity: float = 0.0) -> dict:
    """Generate a single sensor reading for a machine."""
    profile = MACHINE_PROFILES.get(machine_id, {'temp_offset': 0, 'vib_offset': 0, 'health': 0.90})

    r = RANGES
    temp_base  = (r['temperature']['min'] + r['temperature']['max']) / 2 + profile['temp_offset']
    press_base = (r['pressure']['min']    + r['pressure']['max'])    / 2
    vib_base   = (r['vibration']['min']   + r['vibration']['max'])   / 2 + profile['vib_offset']
    hum_base   = (r['humidity']['min']    + r['humidity']['max'])    / 2
    spd_base   = (r['speed']['min']       + r['speed']['max'])       / 2
    rate_base  = (r['production_rate']['min'] + r['production_rate']['max']) / 2

    noise = lambda scale: random.gauss(0, scale)

    if anomaly_mode:
        # escalate values outside normal range based on severity (0..1)
        temperature     = temp_base  + noise(1) + severity * 18
        pressure        = press_base + noise(0.15) + random.choice([-1, 1]) * severity * 2.5
        vibration       = vib_base   + noise(0.1) + severity * 4.0
        humidity        = hum_base   + noise(2)
        speed           = spd_base   + noise(20)  - severity * 100
        production_rate = rate_base  + noise(3)   - severity * 20
    else:
        temperature     = temp_base  + noise(1.5)
        pressure        = press_base + noise(0.2)
        vibration       = vib_base   + noise(0.2)
        humidity        = hum_base   + noise(3)
        speed           = spd_base   + noise(30)
        production_rate = rate_base  + noise(4)

    # Quality score: drops as deviations increase
    temp_dev  = max(0, abs(temperature - temp_base)  / (r['temperature']['max'] - temp_base))
    vib_dev   = max(0, abs(vibration   - vib_base)   / (r['vibration']['max']   - vib_base))
    press_dev = max(0, abs(pressure    - press_base) / (r['pressure']['max']    - press_base))
    deviation = temp_dev * 0.45 + vib_dev * 0.35 + press_dev * 0.20
    quality   = _clamp(100 - deviation * 80 + noise(2), 40, 100)

    return {
        'timestamp':      datetime.utcnow().isoformat(),
        'temperature':    round(_clamp(temperature, 50, 115), 2),
        'pressure':       round(_clamp(pressure, 2.0, 10.0), 3),
        'vibration':      round(_clamp(vibration, 0, 12), 3),
        'humidity':       round(_clamp(humidity, 20, 90), 1),
        'speed':          round(_clamp(speed, 400, 1600), 0),
        'production_rate':round(_clamp(production_rate, 30, 140), 1),
        'quality_score':  round(quality, 1),
    }


def generate_historical_data(machines=None, records_per_machine=150):
    """Generate historical sensor + production data for seeding the DB."""
    if machines is None:
        machines = get_all_machines()

    inserted = 0
    now = datetime.utcnow()

    for machine in machines:
        mid = machine['id']
        profile = MACHINE_PROFILES.get(mid, {'health': 0.90})

        for i in range(records_per_machine):
            age_hours = records_per_machine - i
            ts = now - timedelta(hours=age_hours)

            # Inject anomalies in last 20% for demo richness
            anomaly = (i > records_per_machine * 0.8) and (random.random() < (1 - profile['health']))
            severity = random.uniform(0.3, 0.8) if anomaly else 0.0

            reading = generate_reading(mid, anomaly_mode=anomaly, severity=severity)
            reading['timestamp'] = ts.isoformat()
            insert_sensor_reading(mid, reading, is_simulated=True)
            inserted += 1

    # Insert some production records
    with get_db() as conn:
        for machine in machines:
            mid = machine['id']
            profile = MACHINE_PROFILES.get(mid, {'health': 0.90})
            for i in range(50):
                ts = now - timedelta(hours=50 - i)
                qty = random.randint(80, 120)
                defect_prob = 1 - profile['health'] + random.uniform(-0.05, 0.15)
                defect_count = int(qty * max(0, min(1, defect_prob)))
                defect_types = ['Surface Defect', 'Dimensional Error', 'Structural Weakness',
                                'Material Flaw', None, None, None]
                conn.execute("""
                    INSERT INTO production_records
                      (machine_id,timestamp,batch_id,quantity,defect_count,defect_rate,defect_type,quality_score,shift,is_simulated)
                    VALUES (?,?,?,?,?,?,?,?,?,1)
                """, (
                    mid, ts.isoformat(),
                    f"BATCH-{mid}-{i:04d}",
                    qty, defect_count,
                    round(defect_count / qty * 100, 2),
                    random.choice(defect_types),
                    round((1 - defect_prob) * 100, 1),
                    random.choice(['Morning', 'Afternoon', 'Night'])
                ))

    return inserted


def get_current_reading(machine_id: str, anomaly_mode: bool = False, severity: float = 0.5) -> dict:
    """Return a current (live) reading for the given machine."""
    return generate_reading(machine_id, anomaly_mode=anomaly_mode, severity=severity)


def get_analytics_data(days: int = 7):
    """Return aggregated analytics data for charts."""
    with get_db() as conn:
        # Daily quality trend
        rows = conn.execute("""
            SELECT DATE(timestamp) as day, AVG(quality_score) as avg_quality,
                   COUNT(*) as records
            FROM sensor_data
            WHERE timestamp >= datetime('now', ?)
            GROUP BY DATE(timestamp)
            ORDER BY day
        """, (f'-{days} days',)).fetchall()
        daily_quality = [{'day': r['day'], 'avg_quality': round(r['avg_quality'] or 0, 1),
                          'records': r['records']} for r in rows]

        # Machine comparison
        rows = conn.execute("""
            SELECT machine_id,
                   AVG(quality_score) as avg_quality,
                   AVG(temperature) as avg_temp,
                   AVG(vibration) as avg_vib,
                   COUNT(*) as records
            FROM sensor_data
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY machine_id
        """).fetchall()
        machine_comparison = [dict(r) for r in rows]

        # Defect rates
        rows = conn.execute("""
            SELECT machine_id, AVG(defect_rate) as avg_defect,
                   SUM(defect_count) as total_defects,
                   SUM(quantity) as total_qty
            FROM production_records
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY machine_id
        """).fetchall()
        defect_rates = [dict(r) for r in rows]

        # Defect types distribution
        rows = conn.execute("""
            SELECT defect_type, COUNT(*) as count
            FROM production_records
            WHERE defect_type IS NOT NULL
            GROUP BY defect_type
            ORDER BY count DESC
        """).fetchall()
        defect_types = [dict(r) for r in rows]

        # Parameter distribution for latest readings
        rows = conn.execute("""
            SELECT temperature, pressure, vibration, humidity, speed, production_rate
            FROM sensor_data
            WHERE timestamp >= datetime('now', '-1 day')
        """).fetchall()
        params = [dict(r) for r in rows]

    return {
        'daily_quality': daily_quality,
        'machine_comparison': machine_comparison,
        'defect_rates': defect_rates,
        'defect_types': defect_types,
        'parameter_samples': params[:200],
    }
