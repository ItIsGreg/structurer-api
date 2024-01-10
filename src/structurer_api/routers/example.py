from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

@router.get("/tutorial/")
async def get_tutorial():
    script_location = Path(__file__).resolve().parent
    file_path = script_location / ".."/ ".."/ ".." / "data" / "example_structurer_tutorial.pdf"
    return FileResponse(file_path, media_type='application/pdf', filename="example_structurer_tutorial.pdf")