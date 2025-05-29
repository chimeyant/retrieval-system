from fastapi import Request,UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

templates = Jinja2Templates(directory="resources/view")

async def index(request: Request):
    return ""

async def upload_document(request: Request):
    return templates.TemplateResponse("upload-dokumen.html",{"request": request, "title": "Upload Dokumen"})

async def upload_document_proses(request: Request, file:UploadFile= File(...)):
    # initial directory folder
    upload_dir = "public/uploads"
    
    # Validasi tipe file (opsional)
    if file.content_type not in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Hanya PDF, TXT, dan DOCX yang diizinkan.")

    # Simpan file ke direktori lokal
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    return templates.TemplateResponse("upload-dokumen.html",{"request": request, "title": "Upload Dokumen", "filename": file.filename, "content-type": file.content_type, "size": os.path.getsize(file_location), "message":"File berhasil diunggah" })