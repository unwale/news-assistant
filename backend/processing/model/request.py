from pydantic import BaseModel


class ProcessNewsRequest(BaseModel):
    content: str


class ProcessQueryRequest(BaseModel):
    text: str
