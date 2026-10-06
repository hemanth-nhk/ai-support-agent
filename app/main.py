from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from strawberry.fastapi import GraphQLRouter

from app.config import BASE_DIR, settings
from app.logging_config import configure_logging
from app.api.rest import router as rest_router
from app.api.documents import router as documents_router
from app.api.graphql import schema

configure_logging()

app = FastAPI(title=settings.app_name)

app.include_router(rest_router)
app.include_router(documents_router)
app.include_router(GraphQLRouter(schema), prefix="/graphql")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


@app.get("/app", include_in_schema=False)
def get_app():
    return FileResponse(BASE_DIR / "static" / "index.html")
