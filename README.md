# Resume ATS Scorer 📄

A complete NLP-powered ATS (Applicant Tracking System) scoring tool that analyzes
resume–job description fit using keyword extraction, TF-IDF cosine similarity,
and structural gap analysis.

## What is an ATS Resume Scorer?
An ATS (Applicant Tracking System) Resume Scorer analyzes how well a resume aligns with a job description using:

- Keyword extraction - Identify required skills & technologies
- Semantic similarity - Measure content overlap (TF-IDF cosine similarity)
- Structural analysis - Detect contact info, education, sections
- Gap identification - Show missing keywords and categories

## Why This Matters
Real-world impact:
- 98% of Fortune 500 companies use ATS systems
- Resumes are filtered by ATS before human review
- Keyword match is primary ATS criterion
- Many qualified candidates are filtered out automatically
- This tool helps candidates optimize before applying

## Key Advantages
✓ Fast - Sub-second scoring with no external API calls 

✓ Interpretable - Show exactly why resume matches or doesn't 

✓ Multi-modal - Accept plain text or .docx files 

✓ Comprehensive - TF-IDF + keyword matching + structural analysis 

✓ Actionable - Provide specific improvement suggestions 

✓ No model download - Uses spaCy blank tokenizer (fast, lightweight)

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
<img width="1408" height="768" alt="unnamed" src="https://github.com/user-attachments/assets/40f899f8-ead9-42aa-9e76-f2a76535f2e2" />

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
