from fastapi import APIRouter,Request
from app.controllers import users_controller

router = APIRouter()

@router.get("/users")
async def user_index(request:Request):
    return users_controller.fetch(request)