# Resume ATS Scorer 📄

A complete NLP-powered ATS (Applicant Tracking System) scoring tool that analyzes
resume–job description fit using keyword extraction, TF-IDF cosine similarity,
and structural gap analysis.

---

## Quick Start

```bash
pip install -r requirements.txt

# Download NLTK data (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# (Optional) Download spaCy model for better NLP
python -m spacy download en_core_web_sm

# Run the server
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** — paste a job description + resume, click **Analyze Resume**.

---

## Architecture

```
Job Description (text / .docx)       Resume (text / .docx)
         │                                    │
         ▼                                    ▼
   clean_text()                         clean_text()
         │                                    │
         ▼                                    │
 extract_keywords()                           │
   ├── Multi-word phrase regex               │
   │   (machine learning, FastAPI…)          │
   ├── spaCy blank tokenizer                 │
   ├── NLTK stopword filtering               │
   └── PorterStemmer normalization           │
         │                                    │
         ▼                                    ▼
  keyword_match_score() ─────────────────────┘
   (stem-based matching, literal substring check)
         │
         ▼
 compute_tfidf_similarity()
   TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
   cosine_similarity(jd_vector, resume_vector)
         │
         ▼
 structural_analysis()
   (email, phone, LinkedIn, education, experience,
    projects, skills section, quantified achievements)
         │
         ▼
 Composite ATS Score
   40% × TF-IDF cosine similarity
   45% × keyword match percentage
   15% × structural completeness
         │
         ▼
 FastAPI JSON Response
   {ats_score, grade, verdict, cosine_sim,
    keyword_match, matched_keywords, missing_keywords,
    categories, struct_indicators, tips}
         │
         ▼
 HTML Dashboard (served at GET /)
```

---

## API Reference

### `POST /score` — JSON body

```json
{
  "jd_text": "We're looking for a Python developer with FastAPI...",
  "resume_text": "Experienced Python engineer with 5 years..."
}
```

### `POST /score-files` — multipart form

Fields: `jd_text`, `resume_text`, `jd_file` (.docx), `resume_file` (.docx)

### Response

```json
{
  "ats_score": 72.4,
  "grade": "B",
  "verdict": "Good match — minor gaps to address",
  "cosine_sim": 31.2,
  "keyword_match": 82.1,
  "struct_score": 87.5,
  "matched_keywords": ["python", "fastapi", "machine learning", "docker"],
  "missing_keywords": ["kubernetes", "airflow"],
  "jd_keywords": ["python", "fastapi", "machine learning", "docker", "kubernetes", "airflow"],
  "categories": {
    "Cloud & Infrastructure": ["kubernetes"],
    "Other Technical": ["airflow"]
  },
  "struct_indicators": {
    "email": true, "phone": true, "linkedin": false,
    "education": true, "experience": true, "projects": true,
    "skills_section": true, "achievements": true
  },
  "tips": ["Include your LinkedIn URL — many ATS systems look for it."],
  "jd_word_count": 248,
  "resume_word_count": 412
}
```

---

## Technical Concepts

### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
tfidf = vectorizer.fit_transform([jd_text, resume_text])
# sublinear_tf=True: replaces raw tf with 1 + log(tf)
# ngram_range=(1,2): captures both words and 2-word phrases
```

### Cosine Similarity
Measures the angle between two TF-IDF document vectors in high-dimensional space:
```
similarity = (A · B) / (|A| × |B|)
```
Range 0→1, where 1 = identical documents.

### Stem-Based Keyword Matching
```python
# "managing" matches "management" via shared stem "manag"
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stem = stemmer.stem("managing")  # → "manag"
```

### Composite Score Formula
```
ATS Score = (0.40 × cosine_sim × 100)
          + (0.45 × keyword_match_pct)
          + (0.15 × struct_score × 100)
```

Weights reflect real ATS priorities: keyword density matters most,
semantic fit matters a lot, and structured sections are a bonus signal.

---

## Keyword Extraction Pipeline

1. **Multi-word phrase detection** via 30+ regex patterns for tech terms
   ("machine learning", "natural language processing", "CI/CD pipelines"…)
2. **Single-word tokenization** via spaCy's blank tokenizer (fast, no model needed)
3. **Stopword removal** via NLTK English stopwords + domain-specific noise words
4. **Stem deduplication**: "developer" and "developers" → same keyword
5. **Frequency ranking**: most frequent terms in the JD are prioritized

---

## spaCy Model Note

The tool uses spaCy's **blank English tokenizer** by default — no model download required.
If you install `en_core_web_sm`, the engine automatically uses it for better NLP
(POS tagging, named entity recognition, noun chunk extraction):

```bash
python -m spacy download en_core_web_sm
```

---

## Project Structure

```
ats_scorer/
├── nlp_engine.py   # NLP pipeline: extraction, TF-IDF, scoring, analysis
├── app.py          # FastAPI server + HTML/CSS/JS frontend (single file)
├── requirements.txt
└── README.md
```
