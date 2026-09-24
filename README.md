# 🏭 FactoryIQ

> **AI-powered manufacturing quality intelligence platform.**

FactoryIQ is an AI-powered manufacturing quality control application that monitors machine data, detects anomalies, predicts defects, identifies possible root causes, and recommends corrective actions.

**Detect → Predict → Explain → Prevent**

---

## 🚀 Features

* 📊 Real-time machine monitoring
* 🚨 Anomaly detection
* 📈 Quality and process analysis
* 🤖 Defect prediction
* 🔍 Root cause analysis
* 🛠️ Corrective recommendations
* 📚 RAG-based knowledge assistant
* 💬 AI Copilot for engineers
* 🏭 Factory simulation mode
* 📉 Quality analytics and charts
* 🔔 Alert management
* 📄 AI-generated quality reports

These are the main capabilities documented in the current project.

---

## 🤖 Multi-Agent Workflow

```text id="q8m7x2"
Sensor Data
     ↓
Process Monitoring
     ↓
Quality Analysis
     ↓
Defect Prediction
     ↓
RAG Knowledge
     ↓
Process Optimization
     ↓
AI Explanation
     ↓
Engineer Dashboard
```

### Agents

* **Monitoring Agent** — Detects anomalies and calculates machine health
* **Quality Agent** — Analyzes stability, drift, and risk
* **Defect Agent** — Predicts defect probability and type
* **Optimization Agent** — Generates corrective recommendations
* **RAG Agent** — Retrieves relevant information from uploaded documents

## The repository implements these as separate agents coordinated through the application workflow.

## 🧠 RAG Knowledge Base

FactoryIQ can use uploaded documents as a knowledge source.

```text id="m2x6p1"
Upload Document
      ↓
Text Extraction
      ↓
Chunking
      ↓
TF-IDF Index
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
AI Response
```

The current implementation uses **TF-IDF vectorization and cosine similarity** to retrieve relevant document chunks and inject the retrieved context into Groq prompts.

---

## 🎬 Demo Mode

FactoryIQ includes a factory simulation for demonstrating the complete AI workflow.

Example flow:

```text id="p7k3r2"
Start Simulation
      ↓
Machine B-202 develops anomalies
      ↓
Anomaly Detection
      ↓
Risk Analysis
      ↓
Defect Prediction
      ↓
Root Cause Analysis
      ↓
Corrective Recommendation
      ↓
Dashboard Alert
```

The documented demo uses **Machine B-202** to demonstrate escalating temperature, vibration, and pressure anomalies.

---

## 🛠️ Tech Stack

| Category         | Technology                |
| ---------------- | ------------------------- |
| Backend          | Python, Flask             |
| AI               | Groq API                  |
| Machine Learning | Scikit-learn              |
| Data Processing  | Pandas, NumPy             |
| Database         | SQLite                    |
| RAG              | TF-IDF, Cosine Similarity |
| Frontend         | HTML, CSS, JavaScript     |
| Charts           | Chart.js                  |
| Icons            | Font Awesome              |

---

## 📂 Project Structure

```text id="r5n2kd"
factoryiq/
│
├── agents/
│   ├── orchestrator.py
│   ├── monitoring_agent.py
│   ├── quality_agent.py
│   ├── defect_agent.py
│   ├── optimization_agent.py
│   ├── rag_agent.py
│   └── copilot_agent.py
│
├── services/
│   ├── groq_client.py
│   └── data_service.py
│
├── models/
│   └── database.py
│
├── templates/
├── static/
│
├── app.py
├── config.py
├── requirements.txt
└── .env.example
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash id="z1h5sw"
git clone https://github.com/Giteshkumar23/Manufacturing-Agent.git
cd Manufacturing-Agent
```

### 2. Install dependencies

```bash id="x7k4nc"
pip install -r requirements.txt
```

### 3. Configure environment

```bash id="e3v9pq"
cp .env.example .env
```

Add your Groq API key:

```env id="w2d8kf"
GROQ_API_KEY=your_api_key
```

### 4. Run the application

```bash id="c9j6vz"
python app.py
```

Open:

```text id="u4n7hs"
http://localhost:5000
```

These setup steps follow the repository's current installation instructions.

---

## 🔑 Environment Variables

| Variable        | Description          |
| --------------- | -------------------- |
| `GROQ_API_KEY`  | Groq API key         |
| `GROQ_MODEL`    | Groq model name      |
| `PORT`          | Application port     |
| `DATABASE_PATH` | SQLite database path |
| `SECRET_KEY`    | Flask secret key     |

The application also supports fallback/deterministic operation when a Groq API key is not provided.

---

## 🔌 API

Some of the main endpoints include:

| Method | Endpoint                | Purpose             |
| ------ | ----------------------- | ------------------- |
| GET    | `/api/dashboard`        | Factory overview    |
| GET    | `/api/machines`         | Machine status      |
| GET    | `/api/machines/<id>`    | Machine analysis    |
| GET    | `/api/predictions`      | Defect predictions  |
| GET    | `/api/alerts`           | Alerts              |
| POST   | `/api/ai/analyze`       | Run AI analysis     |
| POST   | `/api/ai/chat`          | AI Copilot          |
| POST   | `/api/documents/upload` | Upload RAG document |
| GET    | `/api/agents/status`    | Agent status        |
| POST   | `/api/simulation/start` | Start simulation    |
| POST   | `/api/simulation/stop`  | Stop simulation     |

The complete endpoint list is documented in the repository.

---

## ⚠️ Important

FactoryIQ is a **decision-support system**, not an industrial control system.

AI-generated recommendations should be reviewed by qualified personnel before being applied to real manufacturing processes. The current demo uses simulated data.

---

## 🔮 Future Improvements

* Real sensor integration using MQTT / OPC-UA
* Advanced vector database for RAG
* ML models trained on production data
* Predictive maintenance
* Multi-factory support
* Real-time WebSocket updates
* PDF report export
* User authentication

---

## 👨‍💻 Author

**Gitesh Kumar Patel**

B.Tech Information Technology

🐙 GitHub: https://github.com/Giteshkumar23

📧 Email: [giteshp321@gmail.com](mailto:giteshp321@gmail.com)

---

## ⭐ Support

If you like this project, consider giving the repository a ⭐.
