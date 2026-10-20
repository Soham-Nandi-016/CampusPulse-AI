# 🎓 CampusPulse AI

> **Multimodal Campus Grievance Redressal & Automated Severity Triage System**
>
> Combines Deep Learning (DistilBERT + MobileNetV2) with Fuzzy Logic (Mamdani FIS) to deliver intelligent, real-time grievance prioritisation.

---

## 🏗️ Architecture

```
CampusPulse AI/
├── app.py                        ← Streamlit entry point (run this)
├── config.py                     ← Centralised settings & domain constants
├── requirements.txt
├── .env.example                  ← Copy to .env and configure
│
├── engines/                      ← ML inference engines (decoupled)
│   ├── nlp_engine.py             ← DistilBERT ABSA + Shannon entropy
│   ├── vision_engine.py          ← MobileNetV2 defect classifier
│   └── fuzzy_engine.py           ← Mamdani FIS (13 rules, centroid defuzz)
│
├── pages/                        ← Streamlit page components
│   ├── student_portal.py         ← Complaint submission form + live AI hints
│   └── admin_dashboard.py        ← Triage board + charts + status dispatch
│
├── data/
│   └── ticket_store.py           ← SQLite persistence + mock data seeder
│
├── models/                       ← Model weight files (after training)
│   ├── nlp/distilbert_absa/      ← HuggingFace checkpoint
│   └── vision/mobilenetv2_defect.pt
│
└── notebooks/
    └── CampusPulse_Model_Training.ipynb  ← Google Colab training pipeline
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set USE_MOCK_MODELS=true for demo (no model weights needed)
```

### 3. Run the app
```bash
streamlit run app.py
```

The app starts with **12 pre-seeded mock tickets** and runs fully in demo mode with no GPU or model weights required.

---

## 🧠 ML Engine Stack

| Engine | Model | Input | Output |
|--------|-------|-------|--------|
| **NLP** | `distilbert-base-uncased` (fine-tuned) | Complaint text | Category probs, urgency [0-1], negativity [0-1], Shannon entropy |
| **Vision** | `MobileNetV2` (transfer learning) | Uploaded image | Defect confidence [0-1], defect label |
| **Fuzzy** | Mamdani FIS (scikit-fuzzy) | Urgency + Negativity + Visual | Priority score [0-100], severity label |

### Fuzzy Rule Summary (13 rules)
| Urgency | Negativity | Visual | → Priority |
|---------|-----------|--------|------------|
| High | High | High | **Critical** |
| High | High | Medium | **Critical** |
| High | Medium | High | High |
| Medium | High | High | High |
| High | High | Low | High |
| Medium | Medium | Medium | Medium |
| Low | Low | Low | Low |
| … | … | … | … |

---

## 🖥️ Admin Dashboard

**Password:** `admin123` (change in `.env`)

Features:
- 6 KPI cards (Total / Critical / High / Open / Resolved / Avg Priority)
- Category × Severity heatmap
- Severity donut chart
- Priority score histogram
- Filterable, sortable triage board (by priority, category, severity, status)
- Per-ticket detail expanders with 1-click status update
- Mamdani membership function visualisation

---

## 🏋️ Training Your Own Models

Open `notebooks/CampusPulse_Model_Training.ipynb` in **Google Colab** (GPU recommended):

1. Run all cells — generates synthetic dataset, fine-tunes DistilBERT, trains MobileNetV2
2. Exports weights to `models/nlp/distilbert_absa/` and `models/vision/mobilenetv2_defect.pt`
3. Copy `models/` folder to the project root
4. Set `USE_MOCK_MODELS=false` in `.env`
5. Restart `streamlit run app.py`

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web framework |
| `transformers` | DistilBERT tokenizer & model |
| `torch` + `torchvision` | Deep learning runtime |
| `scikit-fuzzy` | Mamdani FIS |
| `plotly` | Interactive dashboard charts |
| `Pillow` | Image processing |
| `sqlitedict` | Ticket persistence |
| `loguru` | Structured logging |

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_MODELS` | `true` | Skip ML models, use heuristics |
| `ENABLE_GPU` | `false` | Use CUDA if available |
| `NLP_MODEL_PATH` | `models/nlp/distilbert_absa` | DistilBERT checkpoint directory |
| `VISION_MODEL_PATH` | `models/vision/mobilenetv2_defect.pt` | MobileNetV2 weights |
| `ADMIN_PASSWORD` | `admin123` | Dashboard login password |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

*Built with ❤️ — CampusPulse AI © 2026*
