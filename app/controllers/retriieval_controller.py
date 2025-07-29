from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
import os, re, fitz, math
from rank_bm25 import BM25Okapi
from collections import Counter
import numpy as np
import random

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

def create_vocabulary(documents):
    """
    Membuat vocabulary dari semua dokumen
    """
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc["tokens"])
    return list(vocabulary)

def create_document_term_matrix(documents, vocabulary):
    """
    Membuat matrix dokumen-term
    """
    vocab_dict = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)))
    
    for i, doc in enumerate(documents):
        for token in doc["tokens"]:
            if token in vocab_dict:
                matrix[i][vocab_dict[token]] += 1
    
    return matrix

def initialize_lda_parameters(num_docs, num_words, num_topics):
    """
    Inisialisasi parameter LDA
    """
    # Distribusi topik untuk setiap dokumen (theta)
    theta = np.random.dirichlet([0.1] * num_topics, num_docs)
    
    # Distribusi kata untuk setiap topik (phi)
    phi = np.random.dirichlet([0.1] * num_words, num_topics)
    
    return theta, phi

def calculate_lda_scores(query, documents, num_topics=3, max_iterations=50):
    """
    Menghitung skor LDA untuk setiap dokumen berdasarkan query.
    Implementasi LDA sederhana tanpa gensim.
    """
    if len(documents) == 0:
        return []
    
    # Buat vocabulary dan matrix dokumen-term
    vocabulary = create_vocabulary(documents)
    doc_term_matrix = create_document_term_matrix(documents, vocabulary)
    
    num_docs = len(documents)
    num_words = len(vocabulary)
    
    # Inisialisasi parameter
    theta, phi = initialize_lda_parameters(num_docs, num_words, num_topics)
    
    # Gibbs sampling untuk LDA
    for iteration in range(max_iterations):
        # Update theta dan phi menggunakan Gibbs sampling
        for doc_idx in range(num_docs):
            for word_idx in range(num_words):
                if doc_term_matrix[doc_idx][word_idx] > 0:
                    # Hitung probabilitas untuk setiap topik
                    probs = np.zeros(num_topics)
                    for topic in range(num_topics):
                        probs[topic] = theta[doc_idx][topic] * phi[topic][word_idx]
                    
                    # Normalisasi
                    if np.sum(probs) > 0:
                        probs = probs / np.sum(probs)
                        
                        # Update berdasarkan probabilitas
                        for topic in range(num_topics):
                            theta[doc_idx][topic] = (theta[doc_idx][topic] + probs[topic]) / 2
                            phi[topic][word_idx] = (phi[topic][word_idx] + probs[topic]) / 2
    
    # Tokenisasi query
    query_tokens = tokenize(query)
    
    # Hitung distribusi topik untuk query
    query_vector = np.zeros(num_words)
    vocab_dict = {word: i for i, word in enumerate(vocabulary)}
    
    for token in query_tokens:
        if token in vocab_dict:
            query_vector[vocab_dict[token]] += 1
    
    # Normalisasi query vector
    if np.sum(query_vector) > 0:
        query_vector = query_vector / np.sum(query_vector)
    
    # Hitung distribusi topik untuk query
    query_topic_dist = np.zeros(num_topics)
    for topic in range(num_topics):
        query_topic_dist[topic] = np.sum(query_vector * phi[topic])
    
    # Normalisasi distribusi topik query
    if np.sum(query_topic_dist) > 0:
        query_topic_dist = query_topic_dist / np.sum(query_topic_dist)
    
    # Hitung skor kemiripan antara query dan setiap dokumen
    results = []
    for i, doc in enumerate(documents):
        # Distribusi topik dokumen
        doc_topic_dist = theta[i]
        
        # Hitung cosine similarity antara distribusi topik
        similarity = np.dot(query_topic_dist, doc_topic_dist)
        norm_query = np.linalg.norm(query_topic_dist)
        norm_doc = np.linalg.norm(doc_topic_dist)
        
        if norm_query > 0 and norm_doc > 0:
            similarity = similarity / (norm_query * norm_doc)
        else:
            similarity = 0
        
        results.append({
            "filename": doc["filename"],
            "score": round(similarity, 4),
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
    lda_results = calculate_lda_scores(query, docs)
    
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
    
    