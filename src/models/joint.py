class Joint:
    id: str
    x: int
    y: int

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def to_dict(self):
        return {"id": self.id, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["x"], data["y"])