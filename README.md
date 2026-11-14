# NLP-Based Plagiarism Checker

A Streamlit web app that compares **two documents** and reports how similar their wording is, using classic NLP: NLTK preprocessing, TF-IDF vectorization and cosine similarity. No deep learning, no external APIs.

> **Scope:** the app compares only the two texts you give it. It does **not** search the internet or any database, and textual similarity is **not definitive proof of plagiarism**.

## Features

- Paste text into two side-by-side text areas, or upload `.txt` files (file names are shown)
- Input validation: empty text, unsupported formats, oversized files, UTF-8 decoding errors
- NLTK preprocessing: lowercase, punctuation removal, tokenization, stopword removal, lemmatization
- TF-IDF vectors and cosine similarity, shown as a percentage
- Similarity %, difference %, progress bar, shared keywords with weights, preprocessed text of both documents
- Low / Moderate / High classification with a **user-adjustable threshold slider**
- Built-in sample document pairs for quick demos
- Clear messages for empty vocabulary, stopword-only documents and failed NLTK downloads

## Technology Stack

| Purpose | Library |
|---|---|
| Language | Python 3.9+ |
| Text preprocessing | NLTK |
| TF-IDF, cosine similarity | scikit-learn |
| Numeric work / tables | NumPy, Pandas |
| Web interface | Streamlit |

## Project Structure

```
Plagiarism_Checker/
├── app.py               # Streamlit UI and workflow
├── preprocessing.py     # Cleaning, tokenization, stopwords, lemmatization, NLTK setup
├── plagiarism.py        # TF-IDF, cosine similarity, keywords, classification
├── requirements.txt
├── README.md
├── INTERVIEW_QA.md      # Interview questions and answers about this project
├── .gitignore
├── .streamlit/config.toml   # Light theme and colour palette
├── samples/             # Sample documents for testing
└── tests/               # Unit and end-to-end tests
```

## How It Works

1. **Input** - text is read from the text areas or from uploaded `.txt` files (an upload takes priority).
2. **Validation** - empty, unsupported, oversized or non-UTF-8 inputs produce a helpful message.
3. **Preprocessing** (`preprocessing.py`) - each document is cleaned and reduced to meaningful base-form words.
4. **Vectorization** (`plagiarism.py`) - one `TfidfVectorizer` is fitted on both documents, giving each a numeric vector.
5. **Cosine similarity** - the cosine of the angle between the two vectors, converted to a percentage.
6. **Analysis** - shared keywords are ranked by average TF-IDF weight and the score is classified using your thresholds.

## NLP Concepts Used

- **Tokenization** - splitting text into words.
- **Stopword removal** - dropping filler words (`the`, `is`, `of`) that say little about content.
- **Lemmatization** - reducing words to dictionary form (`studies` -> `study`, `running` -> `run`) so inflected forms match. Each word is tried as a verb, then as a noun, which avoids needing a part-of-speech tagger.
- **TF-IDF** - weights a word by how often it appears in a document (TF) and how rare it is across documents (IDF), so distinctive words count more than common ones. With only two documents the IDF effect is modest: words found in both get the lowest weight, words found in one get a slightly higher one.
- **Cosine similarity** - compares the *direction* of two vectors, so document length matters little. The score runs from 0% (no shared vocabulary) to 100% (identical word weighting).

## Installation (Windows PowerShell)

```powershell
cd Plagiarism_Checker
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once for the current window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Running the App

```powershell
streamlit run app.py
```

The browser opens at `http://localhost:8501`. On first start the app downloads the NLTK `stopwords` and `wordnet` data automatically (internet needed once). To fetch it manually instead:

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running the Tests

```powershell
python -m unittest discover -s tests -v
```

| # | Scenario | Expected behaviour |
|---|---|---|
| 1 | Two identical documents | 100% similarity, 0% difference, High Similarity |
| 2 | Two completely different documents | Almost no shared vocabulary; score below 10% |
| 3 | Two highly similar documents | A few words changed; score stays high (above 70%) but below 100% |
| 4 | Paraphrased sentences | Lower than a lightly edited copy and above unrelated text. TF-IDF compares words, not meaning, so heavy paraphrasing scores low |
| 5 | Empty or invalid documents | A clear error message (empty, whitespace, symbols only, or stopwords only); no crash and no fake score |

`tests/test_plagiarism.py` tests the similarity logic alone (needs only scikit-learn). `tests/test_pipeline.py` runs the five scenarios end to end through NLTK, using the files in `samples/`.

## Sample Documents

`samples/` contains `original.txt`, `similar.txt`, `paraphrased.txt`, `different.txt` and `stopwords_only.txt`. In the app, pick a pair in the sidebar and click **Load sample**.

## Screenshots

Run the app, take your own screenshots and save them in a `screenshots/` folder, then reference them here:

```markdown
![Home screen](screenshots/home.png)
![Results dashboard](screenshots/results.png)
```

Suggested shots: the empty two-column input, a results view with a High Similarity score, and the preprocessed-text expander.

## Limitations

- Compares only two supplied documents; it cannot find sources on the internet.
- Measures shared **wording**, not meaning. Paraphrased or translated copying can score low, and shared technical vocabulary can inflate scores for honest writing.
- English only; text in non-Latin scripts is dropped during cleaning.
- Plain `.txt` input only (no PDF or Word files).
- With just two documents TF-IDF's IDF weighting is weak, so the score behaves much like a weighted word-overlap measure.
- The default 40% / 70% thresholds are adjustable rules of thumb, not validated cut-offs. No accuracy figures are claimed.

## Future Improvements

- Support PDF and DOCX uploads
- Sentence-level matching that highlights the overlapping passages
- Word n-grams (phrases) to catch copied sequences more precisely
- Comparing one document against a folder of documents
- Downloadable PDF/CSV report
- Multi-language support
