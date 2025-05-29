from fastapi import Request,UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

templates = Jinja2Templates(directory="resources/view")

async def index(request: Request):
    folder = "public/uploads"
    files = []
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        if os.path.isfile(full_path):
            size_kb = os.path.getsize(full_path) / 1024
            files.append({
                "name": filename,
                "size_kb": f"{size_kb:.2f}"
            })

    return templates.TemplateResponse("document.html",{"request":request, "files": files})

async def form_upload(request: Request):
    return templates.TemplateResponse("upload-dokumen.html",{"request": request, "title": "Upload Dokumen"})

async def upload_document_proses(request: Request, file:UploadFile= File(...)):
    # initial directory folder
    upload_dir = "public/uploads"
    folder = "public/uploads"
    
    # Validasi tipe file (opsional)
    if file.content_type not in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Hanya PDF, TXT, dan DOCX yang diizinkan.")

    # Simpan file ke direktori lokal
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    files = []
    for filename in os.listdir(folder):
        full_path = os.path.join(folder, filename)
        if os.path.isfile(full_path):
            size_kb = os.path.getsize(full_path) / 1024
            files.append({
                "name": filename,
                "size_kb": f"{size_kb:.2f}"
            })

    
    return templates.TemplateResponse("document.html",{"request": request, "title": "Upload Dokumen", "filename": file.filename, "content-type": file.content_type, "size": os.path.getsize(file_location), "message":"File berhasil diunggah", "files": files })