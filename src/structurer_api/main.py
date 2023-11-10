from fastapi import APIRouter, FastAPI
from structurer_api.routers import structurer
from structurer_api.routers import parse_input
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

router = APIRouter()

router.include_router(
    structurer.router,
    prefix="/structurer",
    tags=["structurer"],
)
router.include_router(
    parse_input.router,
    prefix="/parse_input",
    tags=["parse_input"],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
