"""
Manufacturing AI Orchestrator
Coordinates all agents and decides which ones to run based on input data.
"""
import time
import logging
from datetime import datetime
from models.database import log_agent, get_all_machines
from agents.monitoring_agent import ProcessMonitoringAgent
from agents.quality_agent import QualityAnalysisAgent
from agents.defect_agent import DefectPredictionAgent
from agents.optimization_agent import ProcessOptimizationAgent
from agents.rag_agent import RAGKnowledgeAgent
from services.groq_client import (
    generate_explanation, analyze_root_cause, chat_with_copilot, is_groq_available
)

logger = logging.getLogger(__name__)

# Singleton agent instances
_monitoring_agent    = ProcessMonitoringAgent()
_quality_agent       = QualityAnalysisAgent()
_defect_agent        = DefectPredictionAgent()
_optimization_agent  = ProcessOptimizationAgent()
_rag_agent           = RAGKnowledgeAgent()

# Agent status tracking
_agent_status = {
    'ProcessMonitoringAgent':   {'status': 'idle', 'last_run': None, 'last_result': 'No data yet'},
    'QualityAnalysisAgent':     {'status': 'idle', 'last_run': None, 'last_result': 'No data yet'},
    'DefectPredictionAgent':    {'status': 'idle', 'last_run': None, 'last_result': 'No data yet'},
    'ProcessOptimizationAgent': {'status': 'idle', 'last_run': None, 'last_result': 'No data yet'},
    'RAGKnowledgeAgent':        {'status': 'idle', 'last_run': None, 'last_result': 'Ready'},
}


def _set_status(agent_name: str, status: str, result: str = ''):
    _agent_status[agent_name] = {
        'status':      status,
        'last_run':    datetime.utcnow().isoformat(),
        'last_result': result,
    }


def run_full_pipeline(machine_id: str, reading: dict, use_ai: bool = True) -> dict:
    """
    Run the full multi-agent pipeline for one machine.

    Flow:
        Sensor Data
        → ProcessMonitoringAgent  (always runs)
        → QualityAnalysisAgent    (always runs)
        → DefectPredictionAgent   (always runs)
        → RAGKnowledgeAgent       (runs when anomaly detected)
        → ProcessOptimizationAgent(runs when anomaly detected or risk >= MEDIUM)
        → Groq explanation        (runs when anomaly detected)
    """
    t0 = time.time()
    pipeline_log = []

    # ── Step 1: Process Monitoring ─────────────────────────────────────────────
    _set_status('ProcessMonitoringAgent', 'running')
    monitoring_result = _monitoring_agent.analyze(machine_id, reading)
    _set_status('ProcessMonitoringAgent', 'active',
                monitoring_result['summary'])
    pipeline_log.append({'step': 'monitoring', 'result': monitoring_result['summary']})

    anomaly_detected = bool(monitoring_result['anomalies'])

    # ── Step 2: Quality Analysis ───────────────────────────────────────────────
    _set_status('QualityAnalysisAgent', 'running')
    quality_result = _quality_agent.analyze(machine_id, reading, monitoring_result)
    _set_status('QualityAnalysisAgent', 'active',
                f"Risk={quality_result['risk_level']}, Quality={quality_result['quality_score']}%")
    pipeline_log.append({'step': 'quality', 'result': quality_result['explanation'][:100]})

    # ── Step 3: Defect Prediction ──────────────────────────────────────────────
    _set_status('DefectPredictionAgent', 'running')
    prediction_result = _defect_agent.predict(machine_id, reading, quality_result, monitoring_result)
    _set_status('DefectPredictionAgent', 'active',
                f"Prob={prediction_result['defect_probability']*100:.1f}%, "
                f"Risk={prediction_result['risk_category']}")
    pipeline_log.append({'step': 'prediction', 'result': prediction_result['explanation'][:100]})

    # ── Step 4: RAG retrieval (if anomaly or risk ≥ MEDIUM) ───────────────────
    rag_context = ''
    rag_sources = []
    risk = quality_result.get('risk_level', 'LOW')
    if anomaly_detected or risk in ('MEDIUM', 'HIGH', 'CRITICAL'):
        _set_status('RAGKnowledgeAgent', 'running')
        query = (
            f"Manufacturing quality issue: {monitoring_result['summary']}. "
            f"Risk: {risk}. Defect type: {prediction_result['predicted_defect']}"
        )
        rag_context, rag_sources = _rag_agent.build_rag_context(query)
        _set_status('RAGKnowledgeAgent', 'active',
                    f"Retrieved {len(rag_sources)} knowledge sources" if rag_sources else "No documents indexed")
        pipeline_log.append({'step': 'rag', 'result': f"Sources: {rag_sources}"})
    else:
        _set_status('RAGKnowledgeAgent', 'idle', 'No retrieval needed — normal conditions')

    # ── Step 5: Optimization / Recommendations ─────────────────────────────────
    recommendations = []
    if anomaly_detected or risk in ('HIGH', 'CRITICAL'):
        _set_status('ProcessOptimizationAgent', 'running')
        recommendations = _optimization_agent.recommend(
            machine_id, monitoring_result, quality_result, prediction_result,
            use_ai=use_ai
        )
        _set_status('ProcessOptimizationAgent', 'active',
                    f"Generated {len(recommendations)} recommendations")
        pipeline_log.append({'step': 'optimization', 'result': f"{len(recommendations)} recs generated"})
    else:
        _set_status('ProcessOptimizationAgent', 'idle', 'Normal operation — no recommendations needed')

    # ── Step 6: AI Explanation (Groq) ─────────────────────────────────────────
    ai_explanation = ''
    if anomaly_detected and use_ai:
        ai_explanation = generate_explanation({
            'machine_id':         machine_id,
            'anomalies':          monitoring_result['anomalies'],
            'quality_score':      quality_result['quality_score'],
            'defect_probability': prediction_result['defect_probability'],
            'rag_context':        rag_context,
        })

    # ── Root Cause Analysis ────────────────────────────────────────────────────
    root_cause = None
    if anomaly_detected:
        history_summary = {
            'quality_score': quality_result['quality_score'],
            'process_stability': quality_result['process_stability'],
            'drift_detected': quality_result.get('drift_detected'),
        }
        root_cause = analyze_root_cause(
            monitoring_result['anomalies'], machine_id, history_summary
        )

    total_duration = int((time.time() - t0) * 1000)
    log_agent('Orchestrator', 'full_pipeline',
              f"Machine {machine_id} — {len(pipeline_log)} steps — {total_duration}ms",
              total_duration, machine_id)

    return {
        'machine_id':        machine_id,
        'reading':           reading,
        'monitoring':        monitoring_result,
        'quality':           quality_result,
        'prediction':        prediction_result,
        'recommendations':   recommendations,
        'rag_context':       rag_context,
        'rag_sources':       rag_sources,
        'ai_explanation':    ai_explanation,
        'root_cause':        root_cause,
        'pipeline_log':      pipeline_log,
        'total_duration_ms': total_duration,
        'groq_available':    is_groq_available(),
        'timestamp':         datetime.utcnow().isoformat(),
    }


def get_agent_status() -> dict:
    return {
        name: {**info, 'name': name}
        for name, info in _agent_status.items()
    }


def chat_copilot(user_message: str, factory_context: dict) -> str:
    """Handle chatbot messages using current factory context + RAG."""
    query = user_message
    rag_context, rag_sources = _rag_agent.build_rag_context(query)

    system_context = (
        f"Active alerts: {factory_context.get('active_alerts', 0)}\n"
        f"Overall quality score: {factory_context.get('avg_quality_score', 'N/A')}%\n"
        f"Machines with issues: {factory_context.get('machines_with_issues', [])}\n"
        f"Highest risk machine: {factory_context.get('highest_risk_machine', 'None')}\n"
        f"Simulation active: {factory_context.get('simulation_active', False)}"
    )
    if rag_sources:
        system_context += f"\nKnowledge sources available: {', '.join(rag_sources)}"

    return chat_with_copilot(user_message, system_context, rag_context)
