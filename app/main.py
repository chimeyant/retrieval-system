from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers.web import router as web
from routers.api import router as api

app = FastAPI()

# Static Route
app.mount("/static", StaticFiles(directory="public"), name="static")

# Web Route
app.include_router(web)

# Api Service Route
app.include_router(api, prefix="/api/v1")



