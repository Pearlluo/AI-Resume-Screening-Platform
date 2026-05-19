# Resume Filter — AI-Powered Resume Screening Portal

A Flask web app that screens resumes from Azure Blob Storage using a three-stage pipeline: keyword filtering → Deepseek pre-screening → GPT-4o mini deep analysis. Built for labour hire, mining, civil, and industrial recruitment.

---

## Screenshots

### Keyword search + AI screening results
![AI Screening Results](screenshots/ai_screening.png)

### De-identified view (for privacy)
![De-identified View](screenshots/deidentified.png)

---

## Pipeline

```mermaid
flowchart TD
    A["☁️ Azure Blob Storage\n1330+ resume text files\ncached locally"]:::azure --> B

    B["🔍 Keyword hard filter\nMust match ALL keywords\nScored: hits ×10 + terms ×3"]:::keyword
    B -->|❌ No match| X["Excluded"]:::excluded
    B -->|Top N results| C

    C["🔒 Anonymizer\nCandidate names stripped\nbefore any API call"]:::anon
    C --> D

    D["🔵 Stage 1 — Deepseek  parallel ×5\nFast pre-screen: quick_score 0–100\nrole_relevance, keyword_coverage"]:::deepseek
    D -->|Top deepseek_top_n default 10| E

    E["🟣 Stage 2 — GPT-4o mini  parallel ×3\nmatch_score, recommendation\nmet/missing requirements\nlicence match, risk flags, evidence"]:::gpt
    E -->|Top gpt_top_n default 5| F

    F["✅ Ranked results\nDeepseek + GPT scores\nHighlighted PDF download"]:::output

    classDef azure   fill:#e8f4f8,stroke:#0078d4,color:#003a6b
    classDef keyword fill:#e6f4ea,stroke:#1d9e75,color:#0a4a2a
    classDef excluded fill:#fde8e8,stroke:#e24b4a,color:#7a1f1f
    classDef anon    fill:#f1f0f0,stroke:#888780,color:#2c2c2a
    classDef deepseek fill:#e8f0fe,stroke:#4285f4,color:#0c2d6b
    classDef gpt     fill:#f0ebff,stroke:#7c3aed,color:#2e1065
    classDef output  fill:#ecfdf5,stroke:#059669,color:#064e3b
```

---

## Features

- **Keyword hard filter** — resumes must contain all specified keywords to proceed
- **Requirement term scoring** — key terms auto-extracted from job description and used for scoring
- **Two-stage AI screening** — Deepseek fast pre-screen → GPT-4o mini deep analysis
- **Privacy by design** — candidate names anonymized before any API call; de-identified UI for demos
- **Highlighted PDF download** — matched keywords and requirement terms highlighted in original PDF
- **Azure Blob Storage** — resumes stored in Azure, cached locally for instant search
- **Parallel AI processing** — Deepseek runs 5 concurrent workers, GPT runs 3

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Storage | Azure Blob Storage |
| PDF processing | PyMuPDF (fitz) |
| AI — Stage 1 | Deepseek API (deepseek-chat) |
| AI — Stage 2 | OpenAI GPT-4o mini |
| Parallelism | concurrent.futures.ThreadPoolExecutor |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ResumeFilter.git
cd ResumeFilter
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
AZURE_BLOB_CONTAINER=resumes

OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini

DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_MODEL=deepseek-chat
```

### 4. Run

```bash
python APP.py
```

Visit `http://localhost:5000`

---

## Usage

1. Enter keywords — e.g. `S26, supervisor` (resumes must contain all)
2. Paste the job requirement description
3. Set how many results to show (0 = all)
4. Click **Run Search** — keyword filter runs instantly from local cache
5. Click **Run AI Screening** — sends top candidates through Deepseek → GPT pipeline
6. Download individual highlighted PDFs or all results as a ZIP

---

## Project Structure

```
ResumeFilter/
├── APP.py                  # Flask routes, search logic, server-side state
├── ai_screener.py          # Two-stage parallel AI screening pipeline
├── anonymizer.py           # Name anonymization before API calls
├── requirements.txt
└── templates/
    └── resumesearch1.html  # Frontend UI
```

---

## Privacy

All candidate names are anonymized before being sent to any external AI API. The original resume data in Azure Blob Storage is never modified. The UI includes a de-identified display mode for demos and portfolio use.

---

## Notes

- On first run, resumes are loaded from Azure Blob and cached to `resume_cache.json`
- Subsequent runs load from local cache for fast startup
- Click **Refresh Cache** to re-sync from Azure after resume updates
- `SEARCH_STATE` is stored in server memory — designed for single-user local deployment
