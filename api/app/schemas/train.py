from pydantic import BaseModel, Field
from typing import Literal

class IrisTrainRequest(BaseModel):
    model_type: Literal["rf", "logreg"] = Field(
        "rf", description="Which Iris model to train: 'rf' or 'logreg'"
    )

class CancerTrainRequest(BaseModel):
    model_type: Literal["bayes", "stub"] = Field(
        "bayes", description="Which Cancer model to train: 'bayes' or 'stub'"
    ) 
