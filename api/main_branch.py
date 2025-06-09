import os, sys, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, constr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Langgraph_Workflow.branch_workflow import run_branch_creation_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Automation API - Branch Creator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"],  allow_headers=["*"],
)

class BranchRequest(BaseModel):
    ticketKey: constr(min_length=1)
    gitUrl: HttpUrl
    baseBranch: constr(min_length=1)

class BranchResponse(BaseModel):
    branch: str
    status: str
    error_message: str

def create_feature_branch(req: BranchRequest) -> BranchResponse:
    payload = {
        "ticket_key": req.ticketKey,
        "git_url": str(req.gitUrl),
        "base_branch": req.baseBranch,
    }
    result = run_branch_creation_workflow(payload)
    if result["status"] == "Failed":
        logger.error(f"Branch creation failed: {result['error_message']}")
        raise HTTPException(status_code=500, detail=result["error_message"])
    return BranchResponse(**result)

@app.post("/api/agentic/sessions/create", response_model=BranchResponse)
def agentic_sessions_create(req: BranchRequest):
    return create_feature_branch(req)

@app.post("/api/proxy/api/agentic/sessions/create", response_model=BranchResponse)
def proxy_agentic_sessions_create(req: BranchRequest):
    return create_feature_branch(req)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "branch-creator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
