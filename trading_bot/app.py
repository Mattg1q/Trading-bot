from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import asyncio
import csv
import io
import os
import uvicorn
import sys
from datetime import datetime, timezone

from main import main_loop, bot_state
from data_handler import DataHandler

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
    return {"status": "success", "new_config": config.model_dump()}

@app.get("/api/download/lob.csv")
async def download_lob_csv():
    data_handler = DataHandler()
    try:
        rows, mid_price = await data_handler.get_lob_snapshot_rows()
    finally:
        await data_handler.close()

    output = io.StringIO()
    fieldnames = [
        "timestamp_utc",
        "symbol",
        "level",
        "bid_price",
        "bid_quantity",
        "ask_price",
        "ask_quantity",
        "mid_price",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    timestamp = datetime.now(timezone.utc).isoformat()
    for row in rows:
        writer.writerow({
            "timestamp_utc": timestamp,
            "symbol": bot_state["ticker"],
            **row,
        })

    filename = f"lob_{bot_state['ticker'].replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(output.getvalue(), media_type="text/csv", headers=headers)

if __name__ == "__main__":
    # Avoid 'Event loop is closed' warnings on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
