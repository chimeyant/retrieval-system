from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
import os, re, fitz, math
from rank_bm25 import BM25Okapi
from collections import Counter
import gensim
from gensim import corpora, models

templates = Jinja2Templates(directory="resources/view")

async def analytic(request:Request):
    return templates.TemplateResponse("retrieval-analytic.html", {"request":request})

def tokenize(text):
    return text.lower().split()

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = " ".join(page.get_text() for page in doc)
    return text

def calculate_tf(term, document_tokens):
    """
    Calculate Term Frequency (TF) for a term in a document
    TF = (number of times term appears in document) / (total number of terms in document)
    """
    if not document_tokens:
        return 0
    term_count = document_tokens.count(term)
    return term_count / len(document_tokens)

def calculate_idf(term, documents):
    """
    Calculate Inverse Document Frequency (IDF) for a term
    IDF = log(total number of documents / number of documents containing the term)
    """
    N = len(documents)
    documents_with_term = sum(1 for doc in documents if term in doc["tokens"])
    
    if documents_with_term == 0:
        return 0
    
    return math.log(N / documents_with_term)

def calculate_tfidf(query, documents):
    """
    Calculate TF-IDF scores for documents based on query
    """
    tokenized_query = tokenize(query)
    
    # Calculate IDF for each query term
    idf_scores = {}
    for term in tokenized_query:
        idf_scores[term] = calculate_idf(term, documents)
    
    results = []
    for doc in documents:
        score = 0.0
        
        for term in tokenized_query:
            # Calculate TF for this term in this document
            tf = calculate_tf(term, doc["tokens"])
            
            # Calculate TF-IDF score for this term
            tfidf = tf * idf_scores[term]
            score += tfidf
        
        results.append({
            "filename": doc["filename"], 
            "score": round(score, 4), 
            "methode": "TF-IDF"
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def calculate_bm25(query, documents, k1=1.5, b=0.75):
    tokenized_query = tokenize(query)
    N = len(documents)
    avgdl = sum(len(doc["tokens"]) for doc in documents) / N

    doc_freqs = {}
    for term in tokenized_query:
        doc_freqs[term] = sum(1 for doc in documents if term in doc["tokens"])

    results = []
    for doc in documents:
        score = 0.0
        dl = len(doc["tokens"])
        freqs = Counter(doc["tokens"])

        for term in tokenized_query:
            f = freqs[term]
            df = doc_freqs[term]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score += idf * ((f * (k1 + 1)) / denom) if denom else 0
        results.append({"filename": doc["filename"], "score": round(score, 4), "methode":"BM25"})

    return sorted(results, key=lambda x: x["score"], reverse=True)

def calculate_lda(query, documents, num_topics=3):
    """
    Menghitung skor LDA untuk setiap dokumen berdasarkan query.
    """
    # Siapkan data
    texts = [doc["tokens"] for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Buat model LDA
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    
    # Tokenisasi query
    query_bow = dictionary.doc2bow(tokenize(query))
    
    # Dapatkan distribusi topik untuk query
    query_topics = lda.get_document_topics(query_bow)
    
    # Hitung kemiripan antara distribusi topik query dan dokumen
    results = []
    for i, doc in enumerate(documents):
        doc_topics = lda.get_document_topics(corpus[i])
        # Skor = jumlah topik yang sama antara query dan dokumen dikali bobotnya
        score = 0.0
        for q_topic, q_weight in query_topics:
            for d_topic, d_weight in doc_topics:
                if q_topic == d_topic:
                    score += q_weight * d_weight
        results.append({
            "filename": doc["filename"],
            "score": round(score, 4),
            "methode": "LDA"
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)

async def analytic_proses(request:Request, query):
    PDF_FOLDER = "public/uploads"
    
    docs = []
    for filename in os.listdir(PDF_FOLDER):
        path = os.path.join(PDF_FOLDER, filename)
        text = extract_text_from_pdf(path)
        tokens = tokenize(text)
        docs.append({"filename": filename, "tokens": tokens})
    
    # Calculate scores using BM25, TF-IDF, dan LDA
    bm25_results = calculate_bm25(query, docs)
    tfidf_results = calculate_tfidf(query, docs)
    lda_results = calculate_lda(query, docs)
    
    # Combine results for display
    combined_results = []
    for i, bm25_result in enumerate(bm25_results):
        tfidf_score = next((r["score"] for r in tfidf_results if r["filename"] == bm25_result["filename"]), 0)
        lda_score = next((r["score"] for r in lda_results if r["filename"] == bm25_result["filename"]), 0)
        combined_results.append({
            "filename": bm25_result["filename"],
            "bm25_score": bm25_result["score"],
            "tfidf_score": tfidf_score,
            "lda_score": lda_score
        })
    
    print("BM25 results:", bm25_results)
    print("TF-IDF results:", tfidf_results)
    print("LDA results:", lda_results)
    
    return templates.TemplateResponse("retrieval-analytic-result.html",{
        "request": request,
        "katakunci": query, 
        "results": combined_results
    })
    
    