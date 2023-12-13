from fastapi import APIRouter
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


es = Elasticsearch(
    "https://localhost:9200",
    http_auth=("elastic", "TpRCg2_ee+x6TW+8*7cu"),
    ca_certs="/home/gregor/Desktop/structurer-api/http_ca.crt",
)


class CodeSystemRequest(BaseModel):
    search_term: str


class CodeSystemResponse(BaseModel):
    codeTerm: str
    code: str
    index: str
    score: float


router = APIRouter()
model = SentenceTransformer("all-mpnet-base-v2")


@router.post("/searchSnomed/")
async def searchSnomed(req: CodeSystemRequest) -> CodeSystemResponse:
    vector_search_term = model.encode(req.search_term)
    query = {
        "field": "embedding",
        "query_vector": vector_search_term,
        "k": 1,
        "num_candidates": 100,
    }
    search_result = es.search(index="snomed", knn=query, source=["term", "conceptId"])[
        "hits"
    ]["hits"][0]
    response = CodeSystemResponse(
        codeTerm=search_result["_source"]["term"],
        code=str(search_result["_source"]["conceptId"]),
        index=search_result["_index"],
        score=search_result["_score"],
    )
    return response


@router.post("/searchICD10/")
async def searchICD10(req: CodeSystemRequest) -> CodeSystemResponse:
    vector_search_term = model.encode(req.search_term)
    query = {
        "field": "DescriptionVector",
        "query_vector": vector_search_term,
        "k": 1,
        "num_candidates": 100,
    }
    search_result = es.search(index="icd10", knn=query, source=["Code", "Description"])[
        "hits"
    ]["hits"][0]
    response = CodeSystemResponse(
        codeTerm=search_result["_source"]["Description"],
        code=str(search_result["_source"]["Code"]),
        index=search_result["_index"],
        score=search_result["_score"],
    )
    return response
