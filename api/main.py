import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, constr
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

app = FastAPI(title="Agentic Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:7007",
        "http://127.0.0.1:7007",
        "http://0.0.0.0:7007"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database models
Base = declarative_base()

class SessionModel(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_key = Column(String, index=True)
    git_url = Column(String)
    base_branch = Column(String)
    prompt = Column(Text)
    status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic schemas
class CreateSessionRequest(BaseModel):
    ticketKey: constr(min_length=1)
    gitUrl: HttpUrl
    baseBranch: constr(min_length=1)
    prompt: str

class CreateSessionResponse(BaseModel):
    session_id: int
    status: str

# Database setup
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/sessions/create", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest, db: Session = Depends(get_db)):
    logger.info(f"Creating session with data: {request}")
    try:
        new_session = SessionModel(
            ticket_key=request.ticketKey,
            git_url=str(request.gitUrl),
            base_branch=request.baseBranch,
            prompt=request.prompt,
            status="Pending",
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        logger.info(f"Session created successfully: {new_session.id}")
        return CreateSessionResponse(session_id=new_session.id, status="Pending")
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add endpoint with full proxy path to handle Backstage proxy
@app.post("/api/proxy/api/agentic/sessions/create", response_model=CreateSessionResponse)
def create_session_with_prefix(request: CreateSessionRequest, db: Session = Depends(get_db)):
    logger.info(f"Creating session with prefix path: {request}")
    return create_session(request, db)

@app.get("/sessions/{session_id}")
def get_session_status(session_id: int, db: Session = Depends(get_db)):
    db_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": db_session.id,
        "ticket_key": db_session.ticket_key,
        "git_url": db_session.git_url,
        "base_branch": db_session.base_branch,
        "prompt": db_session.prompt,
        "status": db_session.status,
        "created_at": db_session.created_at,
        "updated_at": db_session.updated_at,
    }

@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "FastAPI is running in WSL!", "status": "healthy"}

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "healthy", "service": "agentic-automation-api", "environment": "WSL"}

# Add a catch-all route to see what paths are being requested
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def catch_all(path: str, request: Request):
    logger.info(f"Catch-all route hit: {request.method} /{path}")
    return {"message": f"Path /{path} not found", "method": request.method}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server in WSL...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
