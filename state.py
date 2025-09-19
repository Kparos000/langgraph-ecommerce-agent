from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import BaseMessage, add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    remaining_steps: str
    memory: str
    schema: str  # JSON string of table schemas
    context: Dict[str, any]  # Rich context dict (date_span, countries, etc.)