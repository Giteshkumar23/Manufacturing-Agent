"""
FactoryIQ — Manufacturing Process Quality Control Agent
Main Flask application entry point.
"""
import os
import json
import time
import random
import threading
import logging
from datetime import datetime, timedelta
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, send_from_directory)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object('config.DevelopmentConfig')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'md', 'doc'}

# ── Database init ──────────────────────────────────────────────────────────────
from models.database import (
    init_db, get_all_machines, get_machine, get_latest_sensor,
    get_sensor_history, get_alerts, update_alert_status,
    get_recommendations, update_recommendation_status,
    get_latest_predictions, get_machine_predictions,
    get_agent_logs, insert_sensor_reading,
    get_setting, set_setting, get_production_stats, get_db,
    log_agent
)

from services.data_service import (
    generate_historical_data, get_current_reading, get_analytics_data
)
from services.groq_client import (
    generate_quality_report, is_groq_available, generate_explanation,
    analyze_root_cause
)
from agents.orchestrator import (
    run_full_pipeline, get_agent_status, chat_copilot
)
from agents.rag_agent import RAGKnowledgeAgent

init_db()
_rag = RAGKnowledgeAgent()

# ── Simulation state ────────────────────────────────────────────────────────────
_sim_thread = None
_sim_lock   = threading.Lock()
_SIM_INTERVAL = 6  # seconds between sim ticks
_DEMO_MACHINE = 'B-202'  # the machine that "breaks" in demo

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_factory_overview():
    """Build a compact factory status dict used by dashboard and chatbot."""
    machines = get_all_machines()
    all_readings = {}
    quality_scores = []
    defect_risks = []
    machines_with_issues = []

    for m in machines:
        reading = get_latest_sensor(m['id'])
        if reading:
            all_readings[m['id']] = reading
            if reading.get('quality_score'):
                quality_scores.append(reading['quality_score'])
        if m['status'] in ('warning', 'critical'):
            machines_with_issues.append(m['id'])

    predictions = get_latest_predictions(limit=len(machines))
    latest_by_machine = {}
    for p in predictions:
        mid = p['machine_id']
        if mid not in latest_by_machine:
            latest_by_machine[mid] = p

    for p in latest_by_machine.values():
        defect_risks.append(p['defect_probability'])

    avg_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 90.0
    avg_defect  = round(sum(defect_risks) / len(defect_risks) * 100, 1) if defect_risks else 5.0

    active_alerts = len([a for a in get_alerts(status='active', limit=500)])

    highest_risk = max(latest_by_machine.items(),
                       key=lambda x: x[1]['defect_probability'],
                       default=(None, {'defect_probability': 0}))

    return {
        'machines':             machines,
        'all_readings':         all_readings,
        'avg_quality_score':    avg_quality,
        'avg_defect_risk':      avg_defect,
        'active_alerts':        active_alerts,
        'machines_with_issues': machines_with_issues,
        'highest_risk_machine': highest_risk[0],
        'machine_count':        len(machines),
        'simulation_active':    get_setting('simulation_active') == '1',
    }


def _build_machine_detail(machine_id: str) -> dict:
    """Build full machine detail data."""
    machine   = get_machine(machine_id)
    reading   = get_latest_sensor(machine_id) or {}
    history   = get_sensor_history(machine_id, limit=60)
    preds     = get_machine_predictions(machine_id, limit=10)
    alerts    = [a for a in get_alerts(limit=50) if a['machine_id'] == machine_id]
    recs      = [r for r in get_recommendations(limit=50) if r['machine_id'] == machine_id]

    # Run pipeline on latest reading for live result
    pipeline = None
    if reading:
        try:
            pipeline = run_full_pipeline(machine_id, reading, use_ai=True)
        except Exception as e:
            logger.error(f"Pipeline error for {machine_id}: {e}")

    return {
        'machine':   machine,
        'reading':   reading,
        'history':   history,
        'preds':     preds,
        'alerts':    alerts[:20],
        'recs':      recs[:10],
        'pipeline':  pipeline,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation background thread
# ══════════════════════════════════════════════════════════════════════════════

def _simulation_loop():
    """Background simulation — updates sensor readings periodically."""
    logger.info("Simulation loop started")
    tick = 0
    while get_setting('simulation_active') == '1':
        try:
            machines = get_all_machines()
            tick += 1
            demo_severity = min(0.85, 0.2 + tick * 0.05)  # escalate over time

            for machine in machines:
                mid = machine['id']
                is_demo_machine = (mid == _DEMO_MACHINE)
                anomaly_mode = is_demo_machine  # B-202 always in anomaly mode during sim

                reading = get_current_reading(
                    mid, anomaly_mode=anomaly_mode,
                    severity=demo_severity if is_demo_machine else 0.0
                )
                insert_sensor_reading(mid, reading, is_simulated=True)

                # Run pipeline for demo machine only (performance)
                if is_demo_machine:
                    run_full_pipeline(mid, reading, use_ai=False)

        except Exception as e:
            logger.error(f"Simulation tick error: {e}")
        time.sleep(_SIM_INTERVAL)
    logger.info("Simulation loop stopped")


# ══════════════════════════════════════════════════════════════════════════════
#  Page Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/machines')
def machines():
    return render_template('machines.html')


@app.route('/machines/<machine_id>')
def machine_detail(machine_id):
    return render_template('machine_detail.html', machine_id=machine_id)


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/predictions')
def predictions():
    return render_template('predictions.html')


@app.route('/root-cause')
def root_cause():
    return render_template('root_cause.html')


@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')


@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html')


@app.route('/copilot')
def copilot():
    return render_template('copilot.html')


@app.route('/reports')
def reports():
    return render_template('reports.html')


@app.route('/agents')
def agents_page():
    return render_template('agents.html')


@app.route('/data')
def production_data():
    return render_template('production_data.html')


# ══════════════════════════════════════════════════════════════════════════════
#  API Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/dashboard')
def api_dashboard():
    overview = _get_factory_overview()
    machines = overview['machines']
    readings = overview['all_readings']

    # Build machine cards with latest data
    machine_cards = []
    for m in machines:
        mid   = m['id']
        r     = readings.get(mid, {})
        preds = get_machine_predictions(mid, limit=1)
        pred  = preds[0] if preds else {}
        machine_cards.append({
            'id':                mid,
            'name':              m['name'],
            'type':              m['type'],
            'location':          m['location'],
            'status':            m['status'],
            'temperature':       r.get('temperature'),
            'pressure':          r.get('pressure'),
            'vibration':         r.get('vibration'),
            'quality_score':     r.get('quality_score'),
            'defect_probability':pred.get('defect_probability', 0),
            'risk_category':     pred.get('risk_category', 'LOW'),
            'last_maintenance':  m.get('last_maintenance'),
        })

    stats = get_production_stats()
    # Compute process stability from quality scores
    all_qs = [r.get('quality_score', 85) for r in overview['all_readings'].values() if r and r.get('quality_score')]
    stability = round(sum(all_qs) / len(all_qs) * 0.95, 1) if all_qs else 85.0
    # Compute production efficiency from production_rate readings
    all_pr = [r.get('production_rate', 97.5) for r in overview['all_readings'].values() if r and r.get('production_rate')]
    from config import Config
    pr_max = Config.NORMAL_RANGES['production_rate']['max']
    efficiency = round(min(100, sum(all_pr) / len(all_pr) / pr_max * 100), 1) if all_pr else 88.0

    return jsonify({
        'kpis': {
            'quality_score':         overview['avg_quality_score'],
            'defect_risk':           overview['avg_defect_risk'],
            'process_stability':     stability,
            'production_efficiency': efficiency,
            'active_alerts':         overview['active_alerts'],
            'machines_monitored':    overview['machine_count'],
        },
        'machines':            machine_cards,
        'simulation_active':   overview['simulation_active'],
        'groq_available':      is_groq_available(),
        'production_stats':    stats,
        'timestamp':           datetime.utcnow().isoformat(),
    })


@app.route('/api/machines')
def api_machines():
    machines = get_all_machines()
    result = []
    for m in machines:
        mid = m['id']
        reading = get_latest_sensor(mid) or {}
        preds   = get_machine_predictions(mid, limit=1)
        pred    = preds[0] if preds else {}
        active_alerts = len([
            a for a in get_alerts(status='active', limit=200)
            if a['machine_id'] == mid
        ])
        result.append({
            **m,
            'temperature':       reading.get('temperature'),
            'pressure':          reading.get('pressure'),
            'vibration':         reading.get('vibration'),
            'humidity':          reading.get('humidity'),
            'speed':             reading.get('speed'),
            'production_rate':   reading.get('production_rate'),
            'quality_score':     reading.get('quality_score'),
            'defect_probability':pred.get('defect_probability', 0),
            'risk_category':     pred.get('risk_category', 'LOW'),
            'active_alerts':     active_alerts,
        })
    return jsonify(result)


@app.route('/api/machines/<machine_id>')
def api_machine_detail(machine_id):
    detail = _build_machine_detail(machine_id)
    if not detail['machine']:
        return jsonify({'error': 'Machine not found'}), 404

    # Serialize pipeline safely
    pipeline = detail.get('pipeline') or {}
    return jsonify({
        'machine':    detail['machine'],
        'reading':    detail['reading'],
        'history':    detail['history'][-30:],
        'predictions':detail['preds'],
        'alerts':     detail['alerts'],
        'recommendations': detail['recs'],
        'monitoring': pipeline.get('monitoring', {}),
        'quality':    pipeline.get('quality', {}),
        'prediction': pipeline.get('prediction', {}),
        'ai_explanation': pipeline.get('ai_explanation', ''),
        'root_cause': pipeline.get('root_cause'),
        'rag_sources': pipeline.get('rag_sources', []),
    })


@app.route('/api/sensors')
def api_sensors():
    machine_id = request.args.get('machine_id')
    limit      = int(request.args.get('limit', 50))
    if machine_id:
        data = get_sensor_history(machine_id, limit=limit)
    else:
        all_data = {}
        for m in get_all_machines():
            all_data[m['id']] = get_latest_sensor(m['id'])
        return jsonify(all_data)
    return jsonify(data)


@app.route('/api/sensors', methods=['POST'])
def api_post_sensor():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    machine_id = data.get('machine_id')
    if not machine_id:
        return jsonify({'error': 'machine_id required'}), 400

    insert_sensor_reading(machine_id, data, is_simulated=False)

    # Run analysis pipeline
    result = run_full_pipeline(machine_id, data, use_ai=True)
    return jsonify({
        'success': True,
        'analysis': {
            'monitoring':  result['monitoring'],
            'quality':     result['quality'],
            'prediction':  result['prediction'],
            'recommendations': result['recommendations'],
        }
    })


@app.route('/api/quality')
def api_quality():
    machines = get_all_machines()
    quality_data = []
    for m in machines:
        reading = get_latest_sensor(m['id']) or {}
        preds   = get_machine_predictions(m['id'], limit=1)
        pred    = preds[0] if preds else {}
        quality_data.append({
            'machine_id':        m['id'],
            'machine_name':      m['name'],
            'quality_score':     reading.get('quality_score', 85),
            'defect_probability':pred.get('defect_probability', 0.05),
            'risk_category':     pred.get('risk_category', 'LOW'),
            'stability':         85,
        })
    return jsonify(quality_data)


@app.route('/api/predictions')
def api_predictions():
    limit = int(request.args.get('limit', 50))
    machine_id = request.args.get('machine_id')
    if machine_id:
        data = get_machine_predictions(machine_id, limit=limit)
    else:
        data = get_latest_predictions(limit=limit)
    return jsonify(data)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    machine_id = data.get('machine_id', 'MANUAL')
    reading    = data.get('reading', data)

    # Ensure we have enough data
    for field in ['temperature', 'pressure', 'vibration']:
        if field not in reading:
            return jsonify({'error': f'Missing field: {field}'}), 400

    result = run_full_pipeline(machine_id, reading, use_ai=True)
    return jsonify({
        'monitoring':     result['monitoring'],
        'quality':        result['quality'],
        'prediction':     result['prediction'],
        'recommendations':result['recommendations'],
        'ai_explanation': result['ai_explanation'],
        'root_cause':     result['root_cause'],
        'rag_sources':    result['rag_sources'],
    })


@app.route('/api/alerts')
def api_alerts():
    status = request.args.get('status')
    limit  = int(request.args.get('limit', 200))
    return jsonify(get_alerts(status=status, limit=limit))


@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def api_alert_acknowledge(alert_id):
    update_alert_status(alert_id, 'acknowledged')
    return jsonify({'success': True})


@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
def api_alert_resolve(alert_id):
    update_alert_status(alert_id, 'resolved')
    return jsonify({'success': True})


@app.route('/api/recommendations')
def api_recommendations():
    status = request.args.get('status')
    limit  = int(request.args.get('limit', 100))
    return jsonify(get_recommendations(status=status, limit=limit))


@app.route('/api/recommendations/<int:rec_id>/acknowledge', methods=['POST'])
def api_rec_acknowledge(rec_id):
    update_recommendation_status(rec_id, 'acknowledged')
    return jsonify({'success': True})


@app.route('/api/recommendations/<int:rec_id>/resolve', methods=['POST'])
def api_rec_resolve(rec_id):
    update_recommendation_status(rec_id, 'resolved')
    return jsonify({'success': True})


@app.route('/api/ai/analyze', methods=['POST'])
def api_ai_analyze():
    data = request.get_json() or {}
    machine_id = data.get('machine_id', 'ALL')
    reading    = data.get('reading') or (get_latest_sensor(machine_id) or {})
    if not reading:
        return jsonify({'error': 'No sensor data available'}), 400

    result = run_full_pipeline(machine_id, reading, use_ai=True)
    return jsonify({
        'success':        True,
        'ai_explanation': result['ai_explanation'],
        'root_cause':     result['root_cause'],
        'monitoring':     result['monitoring'],
        'quality':        result['quality'],
        'prediction':     result['prediction'],
        'recommendations':result['recommendations'],
        'rag_sources':    result['rag_sources'],
        'groq_available': result['groq_available'],
    })


@app.route('/api/ai/chat', methods=['POST'])
def api_ai_chat():
    data    = request.get_json() or {}
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    overview = _get_factory_overview()
    response = chat_copilot(message, {
        'avg_quality_score':    overview['avg_quality_score'],
        'active_alerts':        overview['active_alerts'],
        'machines_with_issues': overview['machines_with_issues'],
        'highest_risk_machine': overview['highest_risk_machine'],
        'simulation_active':    overview['simulation_active'],
    })
    return jsonify({
        'response':     response,
        'rag_used':     bool(data.get('rag', True)),
        'groq_available': is_groq_available(),
        'timestamp':    datetime.utcnow().isoformat(),
    })


@app.route('/api/documents/upload', methods=['POST'])
def api_documents_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(f.filename):
        return jsonify({'error': f'File type not allowed. Use: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    content = RAGKnowledgeAgent.extract_text(filepath, f.filename)
    if not content.strip():
        return jsonify({'error': 'Could not extract text from file'}), 400

    doc_type = request.form.get('doc_type', 'manual')
    doc_id   = _rag.ingest_document(filename, f.filename, content, doc_type)

    return jsonify({
        'success':     True,
        'doc_id':      doc_id,
        'filename':    f.filename,
        'char_count':  len(content),
        'message':     f'Document "{f.filename}" ingested successfully.',
    })


@app.route('/api/documents')
def api_documents():
    return jsonify(_rag.get_documents())


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def api_document_delete(doc_id):
    _rag.delete_document(doc_id)
    return jsonify({'success': True})


@app.route('/api/agents/status')
def api_agents_status():
    return jsonify(get_agent_status())


@app.route('/api/agents/logs')
def api_agents_logs():
    agent = request.args.get('agent')
    limit = int(request.args.get('limit', 50))
    return jsonify(get_agent_logs(agent, limit))


@app.route('/api/generate-demo-data', methods=['POST'])
def api_generate_demo_data():
    machines = get_all_machines()
    count = generate_historical_data(machines, records_per_machine=120)
    return jsonify({
        'success': True,
        'message': f'Generated {count} historical sensor readings across {len(machines)} machines.',
        'machines': len(machines),
        'records':  count,
    })


@app.route('/api/simulation/start', methods=['POST'])
def api_simulation_start():
    global _sim_thread
    set_setting('simulation_active', '1')
    with _sim_lock:
        if _sim_thread is None or not _sim_thread.is_alive():
            _sim_thread = threading.Thread(target=_simulation_loop, daemon=True)
            _sim_thread.start()
    log_agent('Orchestrator', 'simulation_start', 'Demo simulation started')
    return jsonify({'success': True, 'message': 'Factory simulation started.'})


@app.route('/api/simulation/stop', methods=['POST'])
def api_simulation_stop():
    set_setting('simulation_active', '0')
    log_agent('Orchestrator', 'simulation_stop', 'Demo simulation stopped')
    return jsonify({'success': True, 'message': 'Factory simulation stopped.'})


@app.route('/api/simulation/status')
def api_simulation_status():
    active = get_setting('simulation_active') == '1'
    return jsonify({'active': active})


@app.route('/api/analytics')
def api_analytics():
    days = int(request.args.get('days', 7))
    data = get_analytics_data(days)
    return jsonify(data)


@app.route('/api/reports/generate', methods=['POST'])
def api_generate_report():
    overview = _get_factory_overview()
    preds    = get_latest_predictions(limit=50)
    avg_def  = 0
    if preds:
        avg_def = sum(p['defect_probability'] for p in preds) / len(preds) * 100

    report_data = {
        'generated_at':  datetime.utcnow().isoformat(),
        'avg_quality':   overview['avg_quality_score'],
        'avg_defect_risk': round(avg_def, 1),
        'active_alerts': overview['active_alerts'],
        'machines':      overview['machine_count'],
        'simulation':    overview['simulation_active'],
    }
    narrative = generate_quality_report(report_data)
    return jsonify({
        'success':   True,
        'report':    report_data,
        'narrative': narrative,
        'groq_used': is_groq_available(),
        'timestamp': datetime.utcnow().isoformat(),
    })


@app.route('/api/upload', methods=['POST'])
def api_upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files accepted here'}), 400

    try:
        import pandas as pd
        import io
        content = f.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        required = ['machine_id', 'temperature']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing required columns: {missing}'}), 400

        inserted = 0
        for _, row in df.iterrows():
            rec = row.to_dict()
            mid = str(rec.get('machine_id', 'UNKNOWN'))
            insert_sensor_reading(mid, rec, is_simulated=False)
            inserted += 1

        return jsonify({
            'success':  True,
            'rows':     inserted,
            'columns':  list(df.columns),
            'message':  f'Imported {inserted} records successfully.',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/root-cause/<machine_id>')
def api_root_cause(machine_id):
    reading = get_latest_sensor(machine_id)
    if not reading:
        return jsonify({'error': 'No data for this machine'}), 404

    from agents.monitoring_agent import ProcessMonitoringAgent
    mon = ProcessMonitoringAgent()
    mon_result = mon.analyze(machine_id, reading)

    if not mon_result['anomalies']:
        return jsonify({'no_anomalies': True, 'machine_id': machine_id})

    history = get_sensor_history(machine_id, limit=30)
    import statistics as st
    history_summary = {}
    for p in ['temperature', 'vibration', 'pressure']:
        vals = [h[p] for h in history if h.get(p) is not None]
        if vals:
            history_summary[p] = {'mean': round(st.mean(vals), 2), 'stdev': round(st.stdev(vals), 3) if len(vals) > 1 else 0}

    rca = analyze_root_cause(mon_result['anomalies'], machine_id, history_summary)
    return jsonify({
        'machine_id': machine_id,
        'anomalies':  mon_result['anomalies'],
        'root_cause': rca,
        'timestamp':  datetime.utcnow().isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Error handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('dashboard.html'), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500
    return render_template('dashboard.html'), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 16 MB)'}), 413


# ══════════════════════════════════════════════════════════════════════════════
#  Startup
# ══════════════════════════════════════════════════════════════════════════════

def _initial_data_check():
    """Seed demo data if DB is empty."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) as c FROM sensor_data").fetchone()['c']
    if count < 10:
        logger.info("Seeding initial demo data...")
        machines = get_all_machines()
        generate_historical_data(machines, records_per_machine=100)
        logger.info("Demo data seeded.")


if __name__ == '__main__':
    _initial_data_check()
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting FactoryIQ on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
