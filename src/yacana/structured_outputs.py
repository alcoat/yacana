from pydantic import BaseModel


class UseOtherTool(BaseModel):
    shouldUseOtherTool: bool


class UseTool(BaseModel):
    useTool: bool


class MakeAnotherToolCall(BaseModel):
    makeAnotherToolCall: bool
