"""Quick test script for FactoryIQ"""
import sys
sys.path.insert(0, '.')
from app import app, _initial_data_check
_initial_data_check()
client = app.test_client()

print('=== FactoryIQ Test Suite ===')

pages = ['/', '/dashboard', '/machines', '/analytics', '/predictions',
         '/root-cause', '/alerts', '/knowledge', '/copilot', '/reports',
         '/agents', '/data', '/machines/B-202']
print('\n--- Page Routes ---')
all_ok = True
for p in pages:
    r = client.get(p)
    ok = r.status_code == 200
    print(f'  {"OK" if ok else "FAIL"} {p}')
    if not ok: all_ok = False

print('\n--- API Endpoints ---')
for path in ['/api/dashboard', '/api/machines', '/api/machines/B-202',
             '/api/quality', '/api/predictions', '/api/alerts',
             '/api/recommendations', '/api/analytics', '/api/agents/status',
             '/api/root-cause/B-202']:
    r = client.get(path)
    ok = r.status_code == 200
    print(f'  {"OK" if ok else "FAIL"} GET {path}')
    if not ok: all_ok = False

print('\n--- Prediction Test ---')
r = client.post('/api/predict', json={
    'machine_id': 'B-202', 'temperature': 97, 'pressure': 3.2,
    'vibration': 5.1, 'humidity': 52, 'speed': 880, 'production_rate': 75
})
data = r.get_json()
pred = data.get('prediction', {})
dp = pred.get('defect_probability', 0)
print(f'  Defect prob: {dp*100:.1f}% Risk: {pred.get("risk_category")}')
print(f'  Defect type: {pred.get("predicted_defect")}')
print(f'  Anomalies: {len(data.get("monitoring", {}).get("anomalies", []))}')
high_risk = dp > 0.5
print(f'  High risk detected: {high_risk}')

print('\n--- Simulation ---')
r = client.post('/api/simulation/start')
print(f'  Start: {r.get_json()["success"]}')
r = client.post('/api/simulation/stop')
print(f'  Stop: {r.get_json()["success"]}')

print('\n--- Demo Data Generation ---')
r = client.post('/api/generate-demo-data')
data = r.get_json()
print(f'  Result: {data["message"][:60]}')

print('\n--- Copilot Chat ---')
r = client.post('/api/ai/chat',
    json={'message': 'What machines have high defect risk?'},
    content_type='application/json')
data = r.get_json()
print(f'  Groq active: {data["groq_available"]}')
print(f'  Response: {data["response"][:100]}...')

print('\n--- Report ---')
r = client.post('/api/reports/generate',
    json={}, content_type='application/json')
data = r.get_json()
print(f'  Success: {data["success"]}')

print('\n=== RESULT:', 'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED', '===')
