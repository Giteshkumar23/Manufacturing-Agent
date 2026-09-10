"""
Agent 2 — Quality Analysis Agent
Analyzes production parameters, detects process drift, calculates quality
score and stability index, and classifies risk level.
"""
import time
import logging
import statistics
from datetime import datetime
from models.database import get_sensor_history, log_agent
from config import Config

logger = logging.getLogger(__name__)
RANGES = Config.NORMAL_RANGES


class QualityAnalysisAgent:
    name = "QualityAnalysisAgent"

    def analyze(self, machine_id: str, current_reading: dict,
                monitoring_result: dict) -> dict:
        """
        Returns:
            {
              'quality_score': float,
              'process_stability': float,
              'risk_level': 'LOW'|'MEDIUM'|'HIGH'|'CRITICAL',
              'drift_detected': bool,
              'drift_parameters': list[str],
              'explanation': str,
              'correlations': list[dict],
            }
        """
        t0 = time.time()

        history = get_sensor_history(machine_id, limit=50)

        # ── Quality score ──────────────────────────────────────────────────────
        q_score = current_reading.get('quality_score', 85.0)
        if history:
            recent_qs = [h['quality_score'] for h in history[-20:]
                         if h['quality_score'] is not None]
            if recent_qs:
                q_score = statistics.mean(recent_qs) * 0.3 + q_score * 0.7

        # ── Process stability ──────────────────────────────────────────────────
        stability = self._calculate_stability(history)

        # ── Drift detection ────────────────────────────────────────────────────
        drift_params = self._detect_drift(history, current_reading)

        # ── Risk level ────────────────────────────────────────────────────────
        num_anomalies = len(monitoring_result.get('anomalies', []))
        critical_count = sum(
            1 for a in monitoring_result.get('anomalies', [])
            if a['severity'] == 'CRITICAL'
        )

        if critical_count >= 2 or q_score < 60 or stability < 40:
            risk = 'CRITICAL'
        elif critical_count >= 1 or q_score < 75 or stability < 60 or drift_params:
            risk = 'HIGH'
        elif num_anomalies >= 1 or q_score < 85 or stability < 75:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        # ── Correlations ──────────────────────────────────────────────────────
        correlations = self._find_correlations(monitoring_result.get('anomalies', []))

        explanation = self._build_explanation(
            machine_id, q_score, stability, risk, drift_params,
            monitoring_result.get('anomalies', []), correlations
        )

        duration = int((time.time() - t0) * 1000)
        summary = f"Quality={q_score:.1f}%, Stability={stability:.1f}%, Risk={risk}"
        log_agent(self.name, 'analyze', summary, duration, machine_id)

        return {
            'quality_score':     round(q_score, 1),
            'process_stability': round(stability, 1),
            'risk_level':        risk,
            'drift_detected':    bool(drift_params),
            'drift_parameters':  drift_params,
            'explanation':       explanation,
            'correlations':      correlations,
        }

    def _calculate_stability(self, history: list) -> float:
        """Stability = 100 minus penalty for high coefficient of variation."""
        if len(history) < 5:
            return 85.0
        params = ['temperature', 'pressure', 'vibration', 'speed']
        penalties = []
        for p in params:
            vals = [h[p] for h in history if h.get(p) is not None]
            if len(vals) < 3:
                continue
            mean = statistics.mean(vals)
            if mean == 0:
                continue
            cv = statistics.stdev(vals) / mean
            penalties.append(cv * 100)
        if not penalties:
            return 85.0
        avg_penalty = statistics.mean(penalties)
        return max(0.0, min(100.0, 100 - avg_penalty * 3.5))

    def _detect_drift(self, history: list, current: dict) -> list:
        """Detect whether any parameter is trending upward/downward over time."""
        if len(history) < 10:
            return []
        drift = []
        params = ['temperature', 'pressure', 'vibration']
        for p in params:
            vals = [h[p] for h in history if h.get(p) is not None]
            if len(vals) < 10:
                continue
            first_half  = statistics.mean(vals[:len(vals)//2])
            second_half = statistics.mean(vals[len(vals)//2:])
            if first_half == 0:
                continue
            change_pct = abs((second_half - first_half) / first_half) * 100
            if change_pct > 8:  # more than 8% drift
                direction = 'increasing' if second_half > first_half else 'decreasing'
                drift.append(f"{p} ({direction}, {change_pct:.1f}%)")
        return drift

    def _find_correlations(self, anomalies: list) -> list:
        """Identify dangerous parameter combinations."""
        params = {a['parameter'] for a in anomalies}
        correlations = []
        if 'temperature' in params and 'vibration' in params:
            correlations.append({
                'parameters': ['temperature', 'vibration'],
                'impact': 'Surface defect and tool wear risk significantly elevated',
                'severity': 'HIGH',
            })
        if 'temperature' in params and 'pressure' in params:
            correlations.append({
                'parameters': ['temperature', 'pressure'],
                'impact': 'Thermal and mechanical stress combination increases structural defect risk',
                'severity': 'HIGH',
            })
        if 'vibration' in params and 'speed' in params:
            correlations.append({
                'parameters': ['vibration', 'speed'],
                'impact': 'Machine resonance condition may cause dimensional inaccuracy',
                'severity': 'MEDIUM',
            })
        return correlations

    def _build_explanation(self, machine_id, q_score, stability, risk,
                           drift_params, anomalies, correlations) -> str:
        parts = [f"Machine {machine_id} quality analysis:"]
        parts.append(f"Quality score is {q_score:.1f}% with process stability at {stability:.1f}%.")
        if anomalies:
            param_list = ', '.join(a['parameter'] for a in anomalies)
            parts.append(f"Out-of-range parameters: {param_list}.")
        if drift_params:
            parts.append(f"Process drift detected in: {'; '.join(drift_params)}.")
        if correlations:
            for c in correlations:
                parts.append(f"Combined {' + '.join(c['parameters'])} deviation: {c['impact']}.")
        parts.append(f"Overall risk assessment: {risk}.")
        return ' '.join(parts)
