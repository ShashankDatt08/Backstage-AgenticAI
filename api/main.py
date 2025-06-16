import os
import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, constr
import threading
import time
from typing import Optional, List, Dict, Any

# Allow import of your workflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Langgraph_Workflow.workflow import run_enhanced_code_generation_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Automation API - GENERIC LangGraph")

# CORS for Backstage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions store
sessions: Dict[int, Dict[str, Any]] = {}
counter = 0

class CreateSessionRequest(BaseModel):
    ticketKey: constr(min_length=1)
    gitUrl: HttpUrl
    baseBranch: constr(min_length=1)
    prompt: str

class SessionStatusResponse(BaseModel):
    session_id: int
    ticket_key: str
    status: str
    current_step: str
    created_at: float
    summary: Optional[str] = None
    changed_files: List[str] = []
    pr_url: Optional[str] = None
    error_message: Optional[str] = None
    branch_name: Optional[str] = None

class CreateSessionResponse(BaseModel):
    session_id: int
    status: str
    message: str

def background_worker(session_id: int):
    """Background worker that executes the workflow and updates session state"""
    try:
        session = sessions[session_id]
        logger.info(f"Starting workflow for session {session_id}")
        
        # Run the workflow with current session data
        results = run_enhanced_code_generation_workflow({
            "session_id": session_id,
            "ticket_key": session["ticket_key"],
            "git_url": session["git_url"],
            "base_branch": session["base_branch"],
            "prompt": session["prompt"]
        })
        
        # Update session with results
        session.update({
            "status": results["status"],
            "current_step": results["current_step"],
            "summary": results.get("summary"),
            "changed_files": results.get("changed_files", []),
            "pr_url": results.get("pr_url"),
            "branch_name": results.get("branch_name")
        })
        
        if "error_message" in results:
            session["error_message"] = results["error_message"]
        
        logger.info(f"Completed workflow for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in workflow for session {session_id}: {str(e)}")
        sessions[session_id].update({
            "status": "Failed",
            "error_message": str(e)
        })

@app.post("/api/proxy/api/agentic/sessions/create", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    global counter
    counter += 1
    session_id = counter

    # Initialize session state
    sessions[session_id] = {
        "session_id": session_id,
        "ticket_key": req.ticketKey,
        "git_url": str(req.gitUrl),
        "base_branch": req.baseBranch,
        "prompt": req.prompt,
        "status": "Pending",
        "current_step": "created",
        "created_at": time.time(),
        "summary": None,
        "changed_files": [],
        "pr_url": None,
        "error_message": None,
        "branch_name": None
    }

    # Start background worker
    thread = threading.Thread(
        target=background_worker,
        args=(session_id,),
        daemon=True
    )
    thread.start()

    return CreateSessionResponse(
        session_id=session_id,
        status="Pending",
        message=f"Session {session_id} started processing"
    )

@app.get("/api/proxy/api/agentic/sessions/{session_id}", response_model=SessionStatusResponse)
def get_session_status(session_id: int):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return SessionStatusResponse(**session)

@app.get("/api/proxy/api/agentic/sessions")
def list_sessions():
    return {
        "sessions": [
            {
                "session_id": sid,
                "ticket_key": data["ticket_key"],
                "status": data["status"],
                "current_step": data["current_step"],
                "created_at": data["created_at"]
            }
            for sid, data in sessions.items()
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "sessions": len(sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)