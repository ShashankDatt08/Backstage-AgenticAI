from pydantic import BaseModel, HttpUrl, constr

class CreateSessionRequest(BaseModel):
    ticketKey: constr(min_length=1)
    gitUrl: HttpUrl
    baseBranch: constr(min_length=1)
    prompt: str

class CreateSessionResponse(BaseModel):
    session_id: int
    status: str
