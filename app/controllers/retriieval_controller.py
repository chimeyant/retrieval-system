from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
import os, re, fitz, math
from rank_bm25 import BM25Okapi
from collections import Counter

templates = Jinja2Templates(directory="resources/view")

async def analytic(request:Request):
    return templates.TemplateResponse("retrieval-analytic.html", {"request":request})

def tokenize(text):
    return text.lower().split()

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = " ".join(page.get_text() for page in doc)
    return text


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
        results.append({"filename": doc["filename"], "score": score, "methode":"BM25"})

    return sorted(results, key=lambda x: x["score"], reverse=True)


async def analytic_proses(request:Request, query):
    PDF_FOLDER = "public/uploads"
    
    docs = []
    for filename in os.listdir(PDF_FOLDER):
        path = os.path.join(PDF_FOLDER, filename)
        text = extract_text_from_pdf(path)
        tokens = tokenize(text)
        docs.append({"filename": filename, "tokens": tokens})
        
    results = calculate_bm25(query, docs)
    
    print("result", results)
    
    return templates.TemplateResponse("retrieval-analytic-result.html",{"request": request,"katakunci": query, "results": results})
    
    