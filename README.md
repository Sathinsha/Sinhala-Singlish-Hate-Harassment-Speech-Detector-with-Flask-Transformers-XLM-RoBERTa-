# 🚫 Hate & Harassment Speech Detection (Sinhala/Singlish)

Transformers-powered, real‑time hate & harassment speech detector with a modern web UI, token‑level (word‑level) analysis, feedback loop, and one‑click incremental fine‑tuning

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Transformers-%23555555?logo=huggingface"/>
  <img src="https://img.shields.io/badge/Backend-Flask-%23000000?logo=flask"/>
  <img src="https://img.shields.io/badge/Model-XLM--RoBERTa-%233b82f6"/>
  <img src="https://img.shields.io/badge/Tokenization-Word--level-%2310b981"/>
  <img src="https://img.shields.io/badge/Status-Active-%2310b981"/>
</p>

---

## ✨ Highlights
- 🧠 XLM‑RoBERTa token‑classification (2 labels: Neutral/Hate)
- 🔤 Word‑level tokenization (no UNK for Sinhala): robust to Sinhala/Singlish
- ⚡ Real‑time web UI (Flask) with modern, polished interface
- 👀 Auto‑hide for HATE outputs + View/Hide toggle
- 🧩 Token‑level (word‑level) explanations & detected hate words
- 📝 Feedback loop (flag false positives/false negatives)
- 🔁 Incremental fine‑tuning on feedback CSV without overwriting base model

---

## 🖥️ Demo (Local)
- Start the server:
  ```bash
  python app.py
  ```
- Open: `http://localhost:5000`
- Try examples (Sinhala/Singlish) and check token highlights + confidence.

> Tip: HATE detections auto‑hide. Click “View” to reveal word‑level details.

---

## 🏗️ Architecture
- `XLM‑RoBERTa` for token classification (Hugging Face Transformers)
- Word‑level tokenization, labels aligned to words
- Flask backend with a modern responsive UI
- Feedback collector → CSV → fine‑tune script → hot‑swap latest snapshot

```
[UI] → /detect → [Inferencer (XLM‑R)] → word predictions + confidence
  └→ /feedback/* → user_flags.csv → retrain_on_feedback.py → new snapshot
         └→ /admin/retrain hot-swaps model (keeps base intact)
```

---

## 📦 Project Structure
```
PRo/
├── app.py                         # Flask app (lazy model init, endpoints)
├── inference_xlm_roberta.py       # Inference (word-level, confidence logic)
├── train_xlm_roberta.py           # Full training (dataset)
├── data_preprocessing_xlm_roberta.py
├── retrain_on_feedback.py         # Incremental fine-tune on feedback CSV
├── config_xlm_roberta.py          # Config + paths (model, feedback)
├── templates/index.html           # Modern UI (sidebar, colorful sections)
├── outputs_xlm_roberta/           # Model/results outputs
├── feedback/user_flags.csv        # Collected feedback (appends)
└── datsets/                       # SOLD dataset & only_hate.csv
```

---

## 🚀 Quick Start

### 1) Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2) Test inference locally
```bash
python inference_xlm_roberta.py
```

### 3) Run the Web App
```bash
python app.py
```
Open `http://localhost:5000`.

---

## 🧪 Training (Full Dataset)
Run a full train (GPU recommended) or use Google Colab:
```bash
python train_xlm_roberta.py
```
This saves the base model to `outputs_xlm_roberta/model`.

> Colab tip: upload `datsets/`, run `train_xlm_roberta.py`, then download the `outputs_xlm_roberta/model` back into this project.

---

## 🔍 Inference Details
- Word‑level tokenization (space split), then convert each word to subtokens; we aggregate confidence per word (max p(hate) across subtokens). 
- Sentence label: HATE if any word is predicted HATE, or if the dictionary triggers.
- Confidence: mean p(hate) over predicted hate words; if dictionary triggers alone → fallback confidence (e.g., 0.8).

---

## 📝 Feedback & Continuous Learning
The UI provides feedback buttons:
- HATE → “Do you think this is NOT HATE?”
- NOT HATE → “Do you think this is HATE?”

Feedback is stored at:
```
feedback/user_flags.csv
# columns:
# timestamp,input_text,model_label,user_thinks_label,user_words,client_ip,user_agent
```

### Fine‑tune on feedback only (keeps base model)
```bash
python retrain_on_feedback.py
```
- Saves to `outputs_xlm_roberta/fine_tuned/<timestamp>`
- Does NOT overwrite `outputs_xlm_roberta/model`

### Hot‑swap to latest fine‑tuned model (from the app)
```bash
curl -X POST http://localhost:5000/admin/retrain \
  -H "Content-Type: application/json" \
  -d '{"epochs":2, "lr":2e-5, "batch_size":8}'
```
Returns the new snapshot path and reloads the in‑memory model.

> Best practice: freeze most layers or use small epochs to avoid drift. You can also mix in a tiny replay subset to mitigate forgetting.

---

## 🔧 Configuration
See `config_xlm_roberta.py`:
- Model & paths (`MODEL_SAVE_PATH`, `RESULTS_SAVE_PATH`)
- Feedback CSV location (`FEEDBACK_CSV`)
- Fine‑tune output base (`RETRAIN_OUTPUT_BASE`)
- Tokenization type (`TOKENIZATION_TYPE = "word_level_by_spaces"`)
- Training hyperparameters

---

## 🌐 REST API
- POST `/detect`
  ```json
  { "text": "#### මෝඩයා" }
  ```
- POST `/feedback/hate_missed` (NOT HATE → actually HATE)
- POST `/feedback/false_positive` (HATE → actually NOT HATE)
- POST `/admin/retrain` (fine‑tune on feedback + swap snapshot)
- GET `/stats`, GET `/health`

---

## 🧾 Dataset
- SOLD train/test JSON (tokens + labels per word)
- `only_hate.csv` dictionary assists rule‑based fallback

> Labels: `0 = Neutral`, `1 = Hate`. The UI highlights hate words and confidence.

---

## 🖼️ UI Preview
- Centered header + status badge
- Left sidebar with smooth‑scroll + active section highlighting
- Colorful section panels (Input / System Overview / Results / Developers)
- Auto‑hide HATE details with View/Hide control
- Inline feedback form (no popups)

---

## 👥 Developers

```
22ug-0722, 22ug1-0634, 22ug1-0360, 22ug1-0763,
22ug1-0754, 22ug1-0802, 21ug1128, 21ug1014, 21ug0990, 21ug1089
```

---

## 📜 License
This repository is for educational and research purposes. Ensure compliance with local regulations regarding hate speech detection and data usage.

---

## 🙌 Acknowledgments
- SOLD Dataset
- Hugging Face Transformers
- XLM‑RoBERTa

> PRs and issues welcome – feel free to open a discussion for feature requests or improvements!

