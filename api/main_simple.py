import os
import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, constr
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph_poc.workflow import run_enhanced_code_generation_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Agentic Automation API - GENERIC LangGraph")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}
counter = 0

class CreateSessionRequest(BaseModel):
    ticketKey: constr(min_length=1)
    gitUrl: HttpUrl
    baseBranch: constr(min_length=1)
    prompt: str

class CreateSessionResponse(BaseModel):
    session_id: int
    status: str
    message: str

@app.post("/api/proxy/api/agentic/sessions/create", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    global counter
    counter += 1
    sid = counter
    sessions[sid] = {
        "session_id": sid,
        "ticket_key": req.ticketKey,
        "git_url": str(req.gitUrl),
        "base_branch": req.baseBranch,
        "prompt": req.prompt,
        "status": "Pending",
        "current_step": "created",
        "created_at": time.time()
    }

    def worker():
        sessions[sid].update(run_enhanced_code_generation_workflow(sessions[sid]))

    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

    return CreateSessionResponse(session_id=sid, status="Pending", message=f"Session {sid} started")

@app.get("/sessions/{session_id}")
def get_status(session_id: int):
    if session_id not in sessions:
        raise HTTPException(404, "Not found")
    return sessions[session_id]

@app.get("/")
def root():
    return {"status":"healthy","sessions":len(sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
