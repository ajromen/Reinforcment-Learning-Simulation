class Generation:
    best_dist: float
    model_id: str

    def __init__(self, best_dist: float, model_id: str):
        self.best_dist = best_dist
        self.model_id = model_id

    def to_dict(self):
        return {"best_dist": self.best_dist, "model_id": self.model_id}

    @classmethod
    def from_dict(cls, data):
        return cls(data["best_dist"], data["model_id"])