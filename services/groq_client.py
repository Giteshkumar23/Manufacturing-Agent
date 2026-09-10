"""
Groq API client — all AI interaction goes through here.
Falls back to deterministic responses when Groq is unavailable.
"""
import os
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed — AI features will use fallback mode.")

_client: Optional[object] = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get('GROQ_API_KEY', '')
    if not api_key or not GROQ_AVAILABLE:
        return None
    try:
        _client = Groq(api_key=api_key)
        return _client
    except Exception as e:
        logger.error(f"Failed to init Groq client: {e}")
        return None


def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> Optional[str]:
    client = _get_client()
    if client is None:
        return None
    model = os.environ.get('GROQ_MODEL', 'llama3-8b-8192')
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return None


# ─── Public API ────────────────────────────────────────────────────────────────

def generate_explanation(context: dict) -> str:
    """Generate a natural-language explanation of a quality/anomaly situation."""
    machine_id = context.get('machine_id', 'Unknown')
    anomalies  = context.get('anomalies', [])
    quality    = context.get('quality_score', 'N/A')
    defect_prob= context.get('defect_probability', 'N/A')

    system = (
        "You are FactoryIQ, an AI manufacturing quality control assistant. "
        "Provide clear, professional explanations of manufacturing process issues "
        "for production engineers. Be concise and technical but understandable."
    )
    user = (
        f"Machine {machine_id} has the following anomalies: {json.dumps(anomalies)}. "
        f"Quality score: {quality}. Defect probability: {defect_prob}. "
        "Explain WHY the quality risk is elevated in 3-4 sentences. "
        "Focus on the relationships between the anomalous parameters and their effect on product quality."
    )
    result = _call_groq(system, user, 400)
    if result:
        return result

    # Fallback
    if not anomalies:
        return (
            f"Machine {machine_id} is operating within normal parameters. "
            "All monitored process variables are within acceptable ranges, "
            "and no quality risks have been identified at this time."
        )
    param_names = [a.get('parameter', '') for a in anomalies]
    return (
        f"Machine {machine_id} is showing elevated quality risk due to out-of-range values in: "
        f"{', '.join(param_names)}. These deviations from normal operating ranges are associated "
        "with increased surface defect probability and dimensional inaccuracy. "
        "Immediate corrective action is recommended to prevent defective product batches. "
        "[AI Fallback — Groq unavailable]"
    )


def analyze_root_cause(anomalies: list, machine_id: str, history_summary: dict) -> dict:
    """Perform root cause analysis given a list of anomalies."""
    system = (
        "You are a root cause analysis expert for manufacturing processes. "
        "Analyze the given process anomalies and identify the most probable root cause."
    )
    user = (
        f"Machine {machine_id} anomalies: {json.dumps(anomalies)}. "
        f"Recent history: {json.dumps(history_summary)}. "
        "Return a JSON object with keys: "
        "'probable_cause' (string), 'confidence' (0-100 int), 'causal_chain' (list of strings), "
        "'evidence' (list of strings). No markdown, only valid JSON."
    )
    result = _call_groq(system, user, 500)
    if result:
        try:
            # Strip potential markdown fences
            clean = result.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
            return json.loads(clean)
        except Exception:
            pass

    # Fallback
    params = [a.get('parameter', 'unknown') for a in anomalies]
    return {
        "probable_cause": f"Simultaneous deviation in {' and '.join(params)} parameters",
        "confidence": 72,
        "causal_chain": [
            f"Anomaly detected in {', '.join(params)}",
            "Parameter interaction increases mechanical stress",
            "Elevated stress leads to surface and dimensional defects"
        ],
        "evidence": [
            f"{a['parameter']}: {a.get('current', 'N/A')} (normal: {a.get('normal_range', 'N/A')})"
            for a in anomalies
        ]
    }


def generate_recommendation(issue: str, machine_id: str, evidence: list) -> dict:
    """Generate a corrective action recommendation."""
    system = (
        "You are a manufacturing process optimization expert. "
        "Provide specific, actionable recommendations for process issues."
    )
    user = (
        f"Machine {machine_id} has the issue: '{issue}'. "
        f"Evidence: {json.dumps(evidence)}. "
        "Return JSON with keys: 'action' (string), 'priority' (HIGH/MEDIUM/LOW), "
        "'expected_impact' (string), 'steps' (list of strings). No markdown."
    )
    result = _call_groq(system, user, 500)
    if result:
        try:
            clean = result.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
            return json.loads(clean)
        except Exception:
            pass

    # Fallback
    return _fallback_recommendation(issue)


def _fallback_recommendation(issue: str) -> dict:
    issue_lower = issue.lower()
    if 'temperature' in issue_lower:
        return {
            "action": "Reduce machine operating temperature to within 70–85°C range",
            "priority": "HIGH",
            "expected_impact": "Reduce surface defect probability by ~60%",
            "steps": [
                "Check coolant flow rate and temperature",
                "Inspect heat exchanger for fouling",
                "Reduce cutting/processing speed by 10%",
                "Monitor temperature for 15 minutes after adjustment"
            ]
        }
    elif 'vibration' in issue_lower:
        return {
            "action": "Inspect machine alignment and bearing condition",
            "priority": "HIGH",
            "expected_impact": "Reduce vibration-induced dimensional errors by ~70%",
            "steps": [
                "Stop machine and perform visual inspection",
                "Check tool/workpiece mounting for looseness",
                "Measure bearing play and replace if excessive",
                "Perform laser alignment check"
            ]
        }
    elif 'pressure' in issue_lower:
        return {
            "action": "Stabilize hydraulic/pneumatic pressure supply",
            "priority": "MEDIUM",
            "expected_impact": "Improve process consistency and reduce pressure-induced defects",
            "steps": [
                "Check pressure regulator calibration",
                "Inspect for leaks in pneumatic lines",
                "Verify pump output pressure",
                "Adjust pressure relief valve setting"
            ]
        }
    else:
        return {
            "action": f"Investigate and correct {issue}",
            "priority": "MEDIUM",
            "expected_impact": "Improve overall process quality score",
            "steps": [
                "Document current observations",
                "Review recent maintenance records",
                "Consult process engineer",
                "Implement corrective action and monitor"
            ]
        }


def chat_with_copilot(user_message: str, system_context: str, rag_context: str = '') -> str:
    """Chat with the FactoryIQ Copilot."""
    system = (
        "You are FactoryIQ Copilot, an AI assistant for manufacturing quality control. "
        "You help production engineers understand process issues, quality risks, and corrective actions. "
        "Always be professional, specific, and data-driven. "
        "IMPORTANT: Always include a disclaimer that AI recommendations must be verified by "
        "qualified engineering personnel before applying changes to industrial equipment.\n\n"
        f"Current Factory Status:\n{system_context}"
    )
    if rag_context:
        system += f"\n\nRelevant Knowledge Base Information:\n{rag_context}"

    result = _call_groq(system, user_message, 600)
    if result:
        return result

    return (
        "I'm currently unable to connect to the AI reasoning engine (Groq API unavailable). "
        "Based on the current system data, I can tell you that the factory is being monitored "
        "and any detected anomalies are shown in the Alerts section. "
        "Please check the Dashboard for the latest machine status and quality indicators. "
        "[FactoryIQ Copilot — Fallback Mode]"
    )


def generate_quality_report(report_data: dict) -> str:
    """Generate a comprehensive quality report narrative."""
    system = (
        "You are a manufacturing quality assurance expert. "
        "Write a professional quality report based on the provided factory data. "
        "Use clear sections: Executive Summary, Key Findings, Critical Issues, Recommendations."
    )
    user = (
        f"Generate a quality report for the following factory data: "
        f"{json.dumps(report_data, default=str)}. "
        "Write in formal engineering report style. Limit to 500 words."
    )
    result = _call_groq(system, user, 700)
    if result:
        return result

    # Fallback narrative
    quality = report_data.get('avg_quality', 'N/A')
    defect_risk = report_data.get('avg_defect_risk', 'N/A')
    active_alerts = report_data.get('active_alerts', 0)
    return (
        f"## Executive Summary\n\n"
        f"The factory is currently operating with an average quality score of {quality}% "
        f"and a defect risk of {defect_risk}%. There are {active_alerts} active alerts requiring attention.\n\n"
        "## Key Findings\n\n"
        "Process monitoring indicates several machines require attention. "
        "Predictive analysis suggests corrective actions should be taken proactively "
        "to prevent quality degradation.\n\n"
        "## Recommendations\n\n"
        "Review all active alerts and implement recommended corrective actions. "
        "Prioritize critical and high-severity items.\n\n"
        "*[Report generated in fallback mode — Groq API unavailable]*"
    )


def is_groq_available() -> bool:
    """Check if Groq API is available."""
    return _get_client() is not None
