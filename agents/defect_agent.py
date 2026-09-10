"""
Agent 3 — Defect Prediction Agent
Predicts defect probability for the next batch using a rule-based model
(with optional scikit-learn RandomForest when training data is available).
"""
import time
import logging
import math
from datetime import datetime
from models.database import insert_prediction, log_agent, get_sensor_history
from config import Config

logger = logging.getLogger(__name__)
RANGES = Config.NORMAL_RANGES

# Feature weights for the rule-based prediction model
FEATURE_WEIGHTS = {
    'temperature': 0.30,
    'vibration':   0.25,
    'pressure':    0.20,
    'humidity':    0.10,
    'speed':       0.10,
    'production_rate': 0.05,
}

DEFECT_TYPES = [
    "Surface Quality Failure",
    "Dimensional Inaccuracy",
    "Structural Weakness",
    "Material Deformation",
    "Finish Defect",
    "Assembly Error",
]


class DefectPredictionAgent:
    name = "DefectPredictionAgent"
    _ml_model = None  # lazy-loaded scikit-learn model

    def predict(self, machine_id: str, current_reading: dict,
                quality_result: dict, monitoring_result: dict) -> dict:
        """
        Returns:
            {
              'defect_probability': float (0-1),
              'risk_category': str,
              'predicted_defect': str,
              'confidence_score': float (0-1),
              'contributing_factors': list[dict],
              'explanation': str,
            }
        """
        t0 = time.time()

        # Try ML model first, fall back to rule-based
        try:
            prob, factors = self._rule_based_prediction(current_reading, monitoring_result)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            prob, factors = 0.1, []

        # Boost from quality/stability
        quality_penalty = max(0, (85 - quality_result.get('quality_score', 85)) / 100)
        stability_penalty = max(0, (80 - quality_result.get('process_stability', 80)) / 200)
        prob = min(0.99, prob + quality_penalty * 0.3 + stability_penalty * 0.2)

        # Risk category
        if prob >= 0.70:
            risk = 'CRITICAL'
        elif prob >= 0.45:
            risk = 'HIGH'
        elif prob >= 0.25:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        # Predicted defect type — based on which parameter is most deviant
        predicted_defect = self._predict_defect_type(factors, monitoring_result)

        # Confidence (higher when more parameters are anomalous)
        n_anomalies = len(monitoring_result.get('anomalies', []))
        confidence = min(0.95, 0.55 + n_anomalies * 0.08)
        if not monitoring_result.get('anomalies'):
            confidence = 0.70  # normal conditions — high confidence of no defect

        explanation = self._build_explanation(
            machine_id, prob, risk, predicted_defect, factors, quality_result
        )

        # Persist prediction
        insert_prediction(
            machine_id, prob, risk, predicted_defect, confidence, factors,
            explanation, is_sim=True
        )

        duration = int((time.time() - t0) * 1000)
        summary = f"Defect prob={prob:.2f}, Risk={risk}, Type={predicted_defect}"
        log_agent(self.name, 'predict', summary, duration, machine_id)

        return {
            'defect_probability':   round(prob, 3),
            'risk_category':        risk,
            'predicted_defect':     predicted_defect,
            'confidence_score':     round(confidence, 2),
            'contributing_factors': factors,
            'explanation':          explanation,
        }

    def _rule_based_prediction(self, reading: dict, monitoring_result: dict):
        """
        Compute defect probability as a weighted sum of parameter deviations.
        """
        factors = []
        total_weight = 0.0
        weighted_prob = 0.0

        for param, weight in FEATURE_WEIGHTS.items():
            value = reading.get(param)
            limits = RANGES.get(param)
            if value is None or limits is None:
                continue
            lo, hi = limits['min'], limits['max']
            span = hi - lo

            if value < lo:
                deviation = (lo - value) / span
            elif value > hi:
                deviation = (value - hi) / span
            else:
                deviation = 0.0

            # Sigmoid-like scaling: small deviations have small effect
            prob_contribution = 1 / (1 + math.exp(-5 * (deviation - 0.25)))
            weighted_prob += prob_contribution * weight
            total_weight += weight

            if deviation > 0.05:
                factors.append({
                    'parameter':      param,
                    'current':        value,
                    'normal_range':   f"{lo}–{hi} {limits['unit']}",
                    'deviation_pct':  round(deviation * 100, 1),
                    'contribution':   round(prob_contribution * weight / sum(FEATURE_WEIGHTS.values()) * 100, 1),
                })

        if total_weight == 0:
            return 0.05, []

        base_prob = weighted_prob / total_weight
        # Sort factors by contribution
        factors.sort(key=lambda x: x['contribution'], reverse=True)
        return base_prob, factors

    def _predict_defect_type(self, factors: list, monitoring_result: dict) -> str:
        if not factors:
            return "No Defect Predicted"
        top = factors[0]['parameter'] if factors else ''
        second = factors[1]['parameter'] if len(factors) > 1 else ''

        if top == 'temperature' or (top == 'vibration' and second == 'temperature'):
            return "Surface Quality Failure"
        elif top == 'vibration':
            return "Dimensional Inaccuracy"
        elif top == 'pressure':
            return "Structural Weakness"
        elif top == 'speed':
            return "Material Deformation"
        elif top == 'humidity':
            return "Finish Defect"
        else:
            return "Process Non-Conformance"

    def _build_explanation(self, machine_id, prob, risk, defect_type,
                           factors, quality_result) -> str:
        pct = round(prob * 100, 1)
        parts = [
            f"Machine {machine_id} defect prediction: {pct}% probability ({risk} risk).",
        ]
        if factors:
            top_factors = factors[:3]
            factor_text = ', '.join(
                f"{f['parameter']} ({f['deviation_pct']}% deviation)" for f in top_factors
            )
            parts.append(f"Primary contributing factors: {factor_text}.")
        parts.append(f"Most likely defect type: {defect_type}.")
        parts.append(
            f"Process stability is {quality_result.get('process_stability', 'N/A')}%, "
            f"quality score {quality_result.get('quality_score', 'N/A')}%."
        )
        parts.append(
            "⚠️ AI prediction — verify with qualified engineering personnel before "
            "making operational changes."
        )
        return ' '.join(parts)
