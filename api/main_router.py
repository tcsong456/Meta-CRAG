from fastapi import FastAPI
from api.open_server_setup import router as open_router
from api.movie_server_setup import router as movie_router
from api.music_server_setup import router as music_router
from api.sports_server_setup import router as sports_router
from api.finance_server_setup import router as finance_router

app = FastAPI(title="kg_api")
app.include_router(open_router)
app.include_router(movie_router)
app.include_router(music_router)
app.include_router(sports_router)
app.include_router(finance_router)

#%%