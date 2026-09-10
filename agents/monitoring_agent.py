"""
Agent 1 — Process Monitoring Agent
Receives sensor data, detects anomalies, calculates process health score,
and generates alerts when parameters leave normal operating ranges.
"""
import time
import logging
from datetime import datetime
from config import Config
from models.database import insert_alert, log_agent, update_machine_status

logger = logging.getLogger(__name__)

RANGES = Config.NORMAL_RANGES

# How much beyond normal range triggers each severity
WARN_FACTOR    = 0.10   # 10% above normal
CRITICAL_FACTOR= 0.25   # 25% above normal


class ProcessMonitoringAgent:
    name = "ProcessMonitoringAgent"

    def analyze(self, machine_id: str, reading: dict) -> dict:
        """
        Analyze a sensor reading for a single machine.

        Returns:
            {
              'machine_id': str,
              'anomalies': list[dict],
              'alerts_generated': list[dict],
              'health_score': float (0-100),
              'status': 'healthy' | 'warning' | 'critical',
              'summary': str,
              'timestamp': str,
            }
        """
        t0 = time.time()
        anomalies = []
        alerts_generated = []

        for param, limits in RANGES.items():
            value = reading.get(param)
            if value is None:
                continue
            lo, hi = limits['min'], limits['max']
            unit    = limits['unit']
            span    = hi - lo

            if value < lo or value > hi:
                deviation = (value - lo) / span if value < lo else (value - hi) / span
                deviation = abs(deviation)

                if deviation > CRITICAL_FACTOR:
                    severity = 'CRITICAL'
                elif deviation > WARN_FACTOR:
                    severity = 'WARNING'
                else:
                    severity = 'WARNING'

                direction = 'above' if value > hi else 'below'
                message = (
                    f"{param.replace('_', ' ').title()} is {direction} normal range: "
                    f"{value} {unit} (normal: {lo}–{hi} {unit})"
                )
                anomaly = {
                    'parameter':    param,
                    'current':      value,
                    'normal_range': f"{lo}–{hi} {unit}",
                    'deviation_pct':round(deviation * 100, 1),
                    'severity':     severity,
                    'direction':    direction,
                    'message':      message,
                }
                anomalies.append(anomaly)

                # Generate DB alert
                insert_alert(
                    machine_id, param, severity,
                    value, lo, hi, message
                )
                alerts_generated.append({'parameter': param, 'severity': severity})

        # Health score: 100 - weighted penalty for each anomaly
        health = 100.0
        for a in anomalies:
            penalty = a['deviation_pct']
            if a['severity'] == 'CRITICAL':
                penalty *= 1.5
            health -= penalty
        health = max(0.0, min(100.0, health))

        # Status
        if health < 50 or any(a['severity'] == 'CRITICAL' for a in anomalies):
            status = 'critical'
        elif health < 75 or anomalies:
            status = 'warning'
        else:
            status = 'healthy'

        update_machine_status(machine_id, status)

        summary = (
            f"No anomalies detected. Health: {health:.1f}%"
            if not anomalies
            else f"{len(anomalies)} anomaly/anomalies detected. Health: {health:.1f}%. "
                 f"Highest severity: {max(a['severity'] for a in anomalies)}"
        )

        duration = int((time.time() - t0) * 1000)
        log_agent(self.name, 'analyze', summary, duration, machine_id)

        return {
            'machine_id':         machine_id,
            'anomalies':          anomalies,
            'alerts_generated':   alerts_generated,
            'health_score':       round(health, 1),
            'status':             status,
            'summary':            summary,
            'timestamp':          reading.get('timestamp', datetime.utcnow().isoformat()),
        }

    def analyze_all(self, machines_data: dict) -> list:
        """Analyze all machines. machines_data = {machine_id: reading}"""
        results = []
        for mid, reading in machines_data.items():
            try:
                results.append(self.analyze(mid, reading))
            except Exception as e:
                logger.error(f"MonitoringAgent error for {mid}: {e}")
        return results
