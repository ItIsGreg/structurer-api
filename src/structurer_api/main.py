from fastapi import APIRouter, FastAPI
from structurer_api.routers import structurer
from structurer_api.routers import parse_input

# from structurer_api.routers import codeSystems
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
# router.include_router(
#     codeSystems.router,
#     prefix="/codeSystems",
#     tags=["codeSystems"],
# )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
