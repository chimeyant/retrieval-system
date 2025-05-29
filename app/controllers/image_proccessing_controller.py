from fastapi import Request,UploadFile, File
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid
from PIL import Image,ImageEnhance


templates = Jinja2Templates(directory="resources/view")

async def restoration(request:Request):
    return  templates.TemplateResponse("restorasi.html",{"request": request, "title":"Image Restoration"})

async def restoration_process(request:Request, file:UploadFile=File(...)):
    upload_dir = "public/uploads"
    os.makedirs(upload_dir,exist_ok=True)
    
     # Ekstensi asli (misal: .jpg, .png)
    _, ext = os.path.splitext(file.filename)
    # Generate nama file unik
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    
    file_path = os.path.join(upload_dir, unique_filename)
    
    
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
        
    
    image = Image.open(file_path)
    
    # Restorasi warna: tingkatkan ketajaman dan kontras
    restored_image = image.convert("RGB")
    sharpness = ImageEnhance.Sharpness(restored_image).enhance(2.0)
    contrast = ImageEnhance.Contrast(sharpness).enhance(1.2)
    color_boosted = ImageEnhance.Color(contrast).enhance(1.3)
    
    
    image_url = f"/static/uploads/{unique_filename}"

    
    try:
        # Simpan hasil restorasi
        restored_filename = f"/restored_{unique_filename}"
        restored_path = os.path.join(upload_dir, restored_filename)
        color_boosted.save(restored_path)
    except Exception as a:
        return templates.TemplateResponse("restorasi.html",{"request": request, "title":"Image Restorasi","message":"Proses tidak berhasil"})
    
    image_gray_url = f"/static/uploads/{restored_filename}"
    
    print(restored_filename)
    
    return templates.TemplateResponse("restorasi.html",{"request": request,"title":"Image Restoration" ,"message":"Proses upload berhasil","image_url": image_url, "image_gray_url": image_gray_url})

# Fungsi Konversi Image Ke Gray Scale
async def convertion(request:Request):
    return templates.TemplateResponse("konversi.html",{"request":request, "title":"Image Konversi"})


async def convertion_proccess(request:Request, file:UploadFile= File(...)):
    upload_dir = "public/uploads"
    os.makedirs(upload_dir,exist_ok=True)
    
     # Ekstensi asli (misal: .jpg, .png)
    _, ext = os.path.splitext(file.filename)
    # Generate nama file unik
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    
    image_url = f"/static/uploads/{unique_filename}"
    
     # Convert ke grayscale
    grayscale_filename = f"gray_{unique_filename}"
    grayscale_path = os.path.join(upload_dir, grayscale_filename)

    
    try:
        image = Image.open(file_path)
        gray_image = image.convert("L")
        gray_image.save(grayscale_path) 
    except Exception as a:
        return templates.TemplateResponse("konversi.html",{"request": request, "title":"Image Restorasi","message":"Proses tidak berhasil"})
    
    image_gray_url = f"/static/uploads/{grayscale_filename}"
    
    return templates.TemplateResponse("konversi.html",{"request": request,"title":"Image Restoration" ,"message":"Proses konversi berhasil","image_url": image_url, "image_gray_url": image_gray_url})
    