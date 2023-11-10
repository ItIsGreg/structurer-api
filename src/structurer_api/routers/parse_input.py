from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from structurer_api.utils.parse_input import extract_pdf_text, extract_scan_text


router = APIRouter()


class ExtractPDFTextRes(BaseModel):
    text: str


@router.post("/extractPDFText/")
async def extractPDFText(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    text = extract_pdf_text(file.file)
    return ExtractPDFTextRes(text=text)


@router.post("/extractScanText/")
async def extractPdfText(file: UploadFile):
    text = extract_scan_text(file.file)
    return ExtractPDFTextRes(text=text)
