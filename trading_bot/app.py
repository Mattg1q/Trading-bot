from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import os
import uvicorn
import sys

from main import main_loop, bot_state

app = FastAPI()

# Mount the static directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    # Run the main loop in the background
    asyncio.create_task(main_loop())

@app.get("/")
def read_root():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/api/state")
def get_state():
    return bot_state

class ConfigUpdate(BaseModel):
    risk_per_trade: float
    poll_interval: int

@app.post("/api/config")
def update_config(config: ConfigUpdate):
    bot_state["risk_per_trade"] = config.risk_per_trade
    bot_state["poll_interval"] = config.poll_interval
    return {"status": "success", "new_config": config.dict()}

if __name__ == "__main__":
    # Avoid 'Event loop is closed' warnings on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
