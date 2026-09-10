# FactoryIQ — AI Manufacturing Process Quality Control Agent

> **Problem Statement No. 37: Manufacturing Process Quality Control Agent**

FactoryIQ is a complete, production-quality AI-powered manufacturing quality intelligence platform that continuously monitors process parameters, detects anomalies, predicts defects, explains root causes, and recommends corrective actions.

---

## 🎯 Problem Statement

Modern manufacturing processes involve dozens of sensor parameters. Small deviations can cause:
- Dimensional inaccuracies
- Surface defects
- Structural weaknesses
- Increased production costs
- Rejected products

Traditional quality control relies on manual inspection and periodic sampling — reactive rather than proactive.

**FactoryIQ solves this with AI:** *Detect → Predict → Explain → Prevent*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-Time Monitoring** | 10 machines, 6 sensor parameters, continuous analysis |
| **Anomaly Detection** | Automatic detection of out-of-range process parameters |
| **Quality Analysis** | Process stability, drift detection, risk classification |
| **Defect Prediction** | AI-powered defect probability and type prediction |
| **Root Cause Analysis** | Automated causal chain analysis with confidence scores |
| **Corrective Recommendations** | Prioritized actions with evidence and expected impact |
| **RAG Knowledge Base** | Upload manuals/SOPs, AI-grounded in your documents |
| **AI Copilot** | Natural language assistant for engineers |
| **Factory Simulation** | Live demo mode with escalating Machine B-202 anomalies |
| **Quality Analytics** | Trend charts, machine comparison, defect distribution |
| **Alert Management** | Acknowledge/resolve workflow for production alerts |
| **Report Generation** | AI-generated quality reports |

---

## 🏗️ Architecture

```
factoryiq/
├── app.py                    # Flask application & API routes
├── config.py                 # Configuration & normal operating ranges
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── orchestrator.py       # Multi-agent coordinator
│   ├── monitoring_agent.py   # Agent 1: Process Monitoring
│   ├── quality_agent.py      # Agent 2: Quality Analysis
│   ├── defect_agent.py       # Agent 3: Defect Prediction
│   ├── optimization_agent.py # Agent 4: Process Optimization
│   ├── rag_agent.py          # Agent 5: RAG Knowledge
│   └── copilot_agent.py      # Copilot wrapper
│
├── services/
│   ├── groq_client.py        # Groq API integration + fallbacks
│   └── data_service.py       # Synthetic data generator + analytics
│
├── models/
│   └── database.py           # SQLite + all DB operations
│
├── templates/                # Jinja2 HTML templates
└── static/                   # CSS + JavaScript
```

---

## 🤖 Multi-Agent Workflow

```
Sensor Data
     ↓
ProcessMonitoringAgent  ← Detects anomalies, health score
     ↓
QualityAnalysisAgent    ← Risk level, stability, drift
     ↓
DefectPredictionAgent   ← Defect probability & type
     ↓ (if anomaly or risk ≥ MEDIUM)
RAGKnowledgeAgent       ← Retrieves relevant knowledge
     ↓
ProcessOptimizationAgent← Corrective recommendations
     ↓
Groq AI Explanation     ← Natural language reasoning
     ↓
Engineer Dashboard      ← Visual display + alerts
```

### Agents

1. **Process Monitoring Agent** — Compares readings against normal ranges (Temp: 70–85°C, Pressure: 4.5–6.5 bar, Vibration: 0.1–2.5 mm/s, etc.), calculates health score, generates alerts.

2. **Quality Analysis Agent** — Analyzes stability via coefficient of variation, detects parameter drift over time, classifies risk (LOW/MEDIUM/HIGH/CRITICAL), identifies parameter correlations.

3. **Defect Prediction Agent** — Weighted deviation model across 6 parameters with sigmoid probability scaling. Predicts defect type (Surface Failure, Dimensional Error, Structural Weakness, etc.).

4. **Process Optimization Agent** — Template-based + Groq-enhanced corrective recommendations with priority, evidence, action steps, and expected impact.

5. **RAG Knowledge Agent** — TF-IDF vectorization of uploaded documents, cosine similarity retrieval, context injection into Groq prompts.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask |
| Database | SQLite (auto-initialized) |
| AI/ML | Groq API (llama3-8b-8192), scikit-learn TF-IDF |
| Data Processing | Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript (ES6+) |
| Charts | Chart.js 4.4 |
| Icons | Font Awesome 6.5 |

---

## 🚀 Installation

### 1. Clone / navigate to project
```bash
cd factoryiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run the application
```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Recommended | Groq API key for AI reasoning |
| `GROQ_MODEL` | Optional | Model name (default: `llama3-8b-8192`) |
| `PORT` | Optional | Server port (default: `5000`) |
| `DATABASE_PATH` | Optional | SQLite file path (default: `factoryiq.db`) |
| `SECRET_KEY` | Optional | Flask secret key |

**Note:** The application works fully in fallback/deterministic mode without a Groq API key.

### Getting a Groq API Key
1. Go to [console.groq.com](https://console.groq.com/)
2. Create a free account
3. Generate an API key
4. Add to `.env`: `GROQ_API_KEY=gsk_...`

---

## 🎬 Demo Mode

The factory simulation is the centerpiece of the competition demo:

1. Click **"Start Simulation"** in the top bar
2. Machine **B-202** begins developing anomalies (temperature ↑, vibration ↑, pressure ↓)
3. Monitoring Agent detects anomalies within 6 seconds
4. Quality Agent classifies risk as HIGH/CRITICAL
5. Defect Agent predicts 70-85% defect probability
6. Optimization Agent generates corrective recommendations
7. Dashboard shows real-time alerts and status changes
8. Visit `/agents` to see all agents' status in real time
9. Ask the Copilot about Machine B-202

### Demo Story (Competition Flow)
```
1. Open FactoryIQ → Show 10 machines, 94%+ quality score
2. Click "Start Simulation"
3. Machine B-202 status turns CRITICAL
4. Go to /machines/B-202 to see full analysis
5. See: anomaly detection → quality analysis → defect prediction
6. View root cause analysis
7. Check recommended actions
8. Ask Copilot: "What is wrong with Machine B-202?"
9. Go to /agents to show multi-agent pipeline
10. Generate quality report
```

---

## 📊 Dataset Format

CSV upload format:
```csv
machine_id,timestamp,temperature,pressure,vibration,humidity,speed,production_rate,quality_score
A-101,2024-01-15T08:00:00,78.5,5.2,1.2,52.0,1050,95.0,92.5
B-202,2024-01-15T08:01:00,96.3,3.8,4.5,54.0,900,80.0,61.2
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard` | GET | Factory overview + KPIs |
| `/api/machines` | GET | All machine status |
| `/api/machines/<id>` | GET | Machine detail + AI analysis |
| `/api/sensors` | GET/POST | Sensor data |
| `/api/predictions` | GET | Defect predictions |
| `/api/predict` | POST | Run prediction on data |
| `/api/alerts` | GET | Alerts list |
| `/api/alerts/<id>/acknowledge` | POST | Acknowledge alert |
| `/api/alerts/<id>/resolve` | POST | Resolve alert |
| `/api/ai/analyze` | POST | Run full AI pipeline |
| `/api/ai/chat` | POST | AI Copilot chat |
| `/api/documents/upload` | POST | Upload RAG document |
| `/api/agents/status` | GET | Agent status |
| `/api/generate-demo-data` | POST | Generate synthetic data |
| `/api/simulation/start` | POST | Start factory simulation |
| `/api/simulation/stop` | POST | Stop simulation |
| `/api/analytics` | GET | Quality analytics data |
| `/api/reports/generate` | POST | Generate quality report |
| `/api/root-cause/<id>` | GET | Root cause analysis |

---

## 📚 RAG System

The RAG Knowledge Agent uses TF-IDF vectorization (via scikit-learn) for document retrieval:

1. Upload documents via `/knowledge` page
2. Text is extracted (PDF via PyPDF2, plain text for TXT/MD)
3. Content is chunked into 400-character overlapping segments
4. TF-IDF index is built in memory
5. At query time, cosine similarity finds top-3 most relevant chunks
6. Retrieved context is injected into Groq prompts

Supported formats: PDF, TXT, MD, DOC

---

## 🔒 Safety Notice

> ⚠️ **FactoryIQ is a decision-support system, NOT a control system.**
>
> All AI recommendations must be reviewed and verified by qualified engineering and quality assurance personnel before any operational changes are made to industrial equipment or processes.
>
> The system uses simulated data for demonstration. Never use simulation data for real production decisions.

---

## 🚀 Future Improvements

- MQTT/OPC-UA real sensor integration
- Full vector database (Pinecone/Chroma) for RAG
- ML model training on uploaded production data
- Predictive maintenance scheduling
- Multi-tenant factory support
- Mobile-responsive PWA
- Real-time WebSocket updates
- PDF export for reports
- User authentication

---

*FactoryIQ — Built for Problem Statement No. 37: Manufacturing Process Quality Control Agent*
