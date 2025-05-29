from fastapi import APIRouter, Request,UploadFile,File, Form
from app.controllers import home_controller, documents_controller, retriieval_controller

router = APIRouter()

@router.get("/")
async def home(request:Request):
    return await home_controller.index(request)

@router.get("/document")
async def documents_index(request:Request):
    return await documents_controller.index(request)

@router.get("/form-upload-document")
async def upload_dokumen(request:Request):
    return await documents_controller.form_upload(request)


@router.post("/upload-dokumen-proses")
async def upload_dokumen_proses(request:Request, file:UploadFile = File(...)):
    return await documents_controller.upload_document_proses(request, file)

@router.get("/retrieval-analytic")
async def retrieval_analytic(request:Request):
    return await retriieval_controller.analytic(request)

@router.post("/retrieval-analytic-proccess")
async def retrieval_analytic_proccess(request:Request, query:str = Form(...)):
    return await retriieval_controller.analytic_proses(request, query)

@router.get("/contact")
async def contact(request:Request):
    return await home_controller.contact(request)
 


