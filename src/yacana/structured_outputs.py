from pydantic import BaseModel


class UseOtherTool(BaseModel):
    shouldUseOtherTool: bool
