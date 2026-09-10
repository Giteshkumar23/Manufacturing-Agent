"""
Agent 4 — Process Optimization Agent
Generates prioritized corrective action recommendations based on detected
anomalies, quality analysis, and defect predictions.
"""
import time
import logging
from datetime import datetime
from models.database import insert_recommendation, log_agent
from services.groq_client import generate_recommendation, _fallback_recommendation

logger = logging.getLogger(__name__)

# Rule-based recommendation templates
TEMPLATES = {
    'temperature': {
        'high': {
            'issue': 'Excessive Operating Temperature',
            'evidence_tpl': 'Current temperature {val}°C vs normal range 70–85°C ({dev:.1f}% above limit)',
            'action': 'Reduce machine operating temperature to 70–85°C range by checking coolant flow, '
                      'inspecting heat exchanger, and reducing processing speed by 10%.',
            'impact': 'Reduce surface defect probability by approximately 55–65%.',
            'priority': 'HIGH',
        },
        'low': {
            'issue': 'Low Operating Temperature',
            'evidence_tpl': 'Current temperature {val}°C is below normal minimum of 70°C',
            'action': 'Investigate heating system and warm-up procedure. Verify thermostat calibration.',
            'impact': 'Restore material processability and dimensional accuracy.',
            'priority': 'MEDIUM',
        },
    },
    'vibration': {
        'high': {
            'issue': 'Excessive Machine Vibration',
            'evidence_tpl': 'Current vibration {val} mm/s vs normal max 2.5 mm/s ({dev:.1f}% above limit)',
            'action': 'Stop machine and inspect: tool/workpiece mounting, bearing condition, '
                      'machine alignment using laser measurement tools.',
            'impact': 'Reduce dimensional inaccuracy defects by approximately 60–70%.',
            'priority': 'HIGH',
        },
    },
    'pressure': {
        'high': {
            'issue': 'Over-Pressure Condition',
            'evidence_tpl': 'Current pressure {val} bar above normal maximum 6.5 bar',
            'action': 'Check pressure relief valve, inspect for blockages in hydraulic lines, '
                      'calibrate pressure control system.',
            'impact': 'Prevent structural weakness defects and equipment damage.',
            'priority': 'HIGH',
        },
        'low': {
            'issue': 'Under-Pressure Condition',
            'evidence_tpl': 'Current pressure {val} bar below normal minimum 4.5 bar',
            'action': 'Inspect pump output, check for leaks in pneumatic/hydraulic lines, '
                      'verify pressure regulator setting.',
            'impact': 'Restore process force consistency and reduce material flow defects.',
            'priority': 'MEDIUM',
        },
    },
    'humidity': {
        'high': {
            'issue': 'High Ambient Humidity',
            'evidence_tpl': 'Current humidity {val}% exceeds recommended 65% maximum',
            'action': 'Check HVAC dehumidification system, inspect for water leaks near machine, '
                      'ensure material storage is properly sealed.',
            'impact': 'Prevent moisture-related material defects and corrosion.',
            'priority': 'MEDIUM',
        },
    },
    'speed': {
        'high': {
            'issue': 'Excessive Machine Speed',
            'evidence_tpl': 'Current speed {val} RPM exceeds normal maximum 1200 RPM',
            'action': 'Reduce machine speed to within 800–1200 RPM range. '
                      'Check drive parameters and speed controller.',
            'impact': 'Reduce tool wear, vibration, and material deformation defects.',
            'priority': 'MEDIUM',
        },
        'low': {
            'issue': 'Below-Specification Machine Speed',
            'evidence_tpl': 'Current speed {val} RPM below normal minimum 800 RPM',
            'action': 'Check motor drive, belt tension, and mechanical load. '
                      'Verify speed setpoint in process controller.',
            'impact': 'Restore production rate and surface finish quality.',
            'priority': 'LOW',
        },
    },
}


class ProcessOptimizationAgent:
    name = "ProcessOptimizationAgent"

    def recommend(self, machine_id: str, monitoring_result: dict,
                  quality_result: dict, prediction_result: dict,
                  use_ai: bool = True) -> list:
        """
        Generate recommendations for all detected anomalies.
        Returns list of recommendation dicts.
        """
        t0 = time.time()
        recommendations = []
        anomalies = monitoring_result.get('anomalies', [])

        for anomaly in anomalies:
            param = anomaly['parameter']
            direction = anomaly.get('direction', 'above')
            value = anomaly.get('current', 0)
            dev = anomaly.get('deviation_pct', 0)

            template = self._get_template(param, direction)
            if template is None:
                continue

            evidence = template['evidence_tpl'].format(val=value, dev=dev)

            # Try AI-enhanced recommendation
            if use_ai:
                ai_rec = generate_recommendation(
                    template['issue'], machine_id,
                    [f"{param}: {value} ({direction} normal range by {dev}%)"]
                )
                action = ai_rec.get('action', template['action'])
                impact = ai_rec.get('expected_impact', template['impact'])
                priority = ai_rec.get('priority', template['priority'])
            else:
                action = template['action']
                impact = template['impact']
                priority = template['priority']

            rec = {
                'machine_id': machine_id,
                'priority':   priority,
                'issue':      template['issue'],
                'evidence':   evidence,
                'action':     action,
                'impact':     impact,
                'parameter':  param,
            }
            recommendations.append(rec)

            # Persist
            insert_recommendation(machine_id, priority, template['issue'],
                                  evidence, action, impact)

        # Add quality-level recommendation if overall risk is high
        if quality_result.get('risk_level') in ('HIGH', 'CRITICAL') and not anomalies:
            rec = {
                'machine_id': machine_id,
                'priority':   'MEDIUM',
                'issue':      'Process Quality Degradation',
                'evidence':   f"Quality score {quality_result.get('quality_score', 'N/A')}%, "
                              f"stability {quality_result.get('process_stability', 'N/A')}%",
                'action':     'Conduct process audit — check tooling condition, material batch, '
                              'and verify all process parameters are within specification.',
                'impact':     'Stabilize quality score and reduce defect risk.',
                'parameter':  'quality',
            }
            recommendations.append(rec)
            insert_recommendation(machine_id, 'MEDIUM', rec['issue'],
                                  rec['evidence'], rec['action'], rec['impact'])

        duration = int((time.time() - t0) * 1000)
        summary = f"Generated {len(recommendations)} recommendations"
        log_agent(self.name, 'recommend', summary, duration, machine_id)

        return recommendations

    def _get_template(self, param: str, direction: str) -> dict | None:
        param_templates = TEMPLATES.get(param)
        if not param_templates:
            return None
        key = 'high' if direction == 'above' else 'low'
        return param_templates.get(key) or list(param_templates.values())[0]
