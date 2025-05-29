from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="resources/view")

async def index(request:Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Beranda"})

async def contact(request:Request):
    return templates.TemplateResponse("contact.html", {"request":request, "title":"Contact US"})