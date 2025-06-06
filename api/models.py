from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    func,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    ticket_key = Column(String(64), nullable=False)
    git_url = Column(String(512), nullable=False)
    base_branch = Column(String(128), nullable=False)
    prompt = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="Pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
