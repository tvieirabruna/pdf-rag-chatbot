from pydantic import BaseModel, Field
from typing_extensions import Annotated, List

class QuestionRequest(BaseModel):
    question: str = Field(description="The user's question to the Chatbot")

class QuestionAnswer(BaseModel):
    answer: str = Field(description="The detailed answer to the question based on the provided context")
    references: Annotated[List[str], ...] = Field(description="The relevant quotes from the context that support the answer")