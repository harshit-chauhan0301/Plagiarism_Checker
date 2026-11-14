# Interview Questions and Answers

**1. Explain your project in one minute.**
It is a Streamlit app that compares two documents. Each text is preprocessed with NLTK (lowercase, punctuation removal, tokenization, stopword removal, lemmatization), converted to TF-IDF vectors with scikit-learn, and compared with cosine similarity. The result is shown as a percentage with shared keywords and a Low/Moderate/High label based on adjustable thresholds. It only compares the two supplied texts; it does not search the internet.

**2. Why did you preprocess the text before vectorizing?**
Raw text contains noise: capital letters, punctuation and filler words. Without cleaning, "Cats" and "cat" would be different features and words like "the" would dominate. Preprocessing makes matching words look identical to the vectorizer.

**3. What is the difference between stemming and lemmatization? Why lemmatization?**
Stemming chops suffixes by rule and can produce non-words ("studies" -> "studi"). Lemmatization uses a dictionary (WordNet) to return a real base form ("studies" -> "study"). Lemmatization is slower but gives cleaner, readable keywords, which matters because the app displays them.

**4. What is TF-IDF?**
Term Frequency is how often a word appears in a document. Inverse Document Frequency lowers the weight of words that appear in many documents. Multiplying them gives high weight to words that are frequent in one document but not everywhere.

**5. Is TF-IDF very useful with only two documents?**
Honestly, its effect is limited. With two documents a word is in either both (lowest IDF) or one (slightly higher IDF), so the IDF part only gives a mild boost to words that are unique to one document. The vectors still give a valid cosine score; it behaves like a weighted word-overlap measure. A larger reference corpus would make IDF more meaningful.

**6. Why cosine similarity instead of Euclidean distance?**
Cosine compares the direction of vectors, not their length, so a long document and a short excerpt of it can still match. Euclidean distance would penalise length differences.

**7. What does a score of 0% or 100% mean?**
0% means the documents share no vocabulary after preprocessing. 100% means identical word weighting; the texts could still differ in word order, because TF-IDF ignores order.

**8. Can your app detect paraphrased plagiarism?**
Mostly not. It compares words, not meaning, so a text rewritten with different words scores low. Detecting that needs semantic models (for example sentence embeddings), which the brief excluded and which I list as a future improvement.

**9. Does it check the internet for plagiarism?**
No. It compares only the two documents supplied. Claiming otherwise would be wrong.

**10. Is a high score proof of plagiarism?**
No. Quotations, shared technical terms and common phrases raise the score. The app says this on screen; a person must review the texts.

**11. How did you choose the 40% and 70% thresholds?**
They are adjustable rules of thumb, not validated values; the user can change them with the slider. I did not tune them against labelled data, and I do not report accuracy figures for that reason.

**12. What edge cases did you handle?**
Empty or whitespace-only input, unsupported file types, files over the size limit, UTF-8 decoding errors, documents with only stopwords or symbols, an empty TF-IDF vocabulary, failed NLTK downloads, and unexpected runtime errors (shown as a friendly message while the details are logged).

**13. Why split the code into three modules?**
`preprocessing.py` cleans text, `plagiarism.py` does the maths, and `app.py` is only the interface. The first two have no Streamlit dependency, so they can be tested and reused on their own, and a change in the UI does not touch the NLP code.

**14. How does the app avoid unsafe HTML?**
User text and file names are passed through `html.escape` before being placed in custom HTML blocks, so pasted content cannot inject markup.

**15. How would you improve it?**
Sentence-level matching with highlighted passages, word n-grams for phrase-level copying, PDF/DOCX input, comparing one document against many, and semantic similarity for paraphrase detection.

**16. How did you test it?**
Unit tests cover the similarity logic with a hand-calculated case (`cat dog bird` vs `cat dog fish` gives about 50.3%). End-to-end tests cover the five required scenarios using the sample documents: identical, different, highly similar, paraphrased, and empty or invalid.
