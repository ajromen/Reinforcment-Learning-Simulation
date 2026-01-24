class Joint:
    id: str
    x: int
    y: int

    def __init__(self, id, x, y, bone_ids):
        self.id = id
        self.x = x
        self.y = y
        self.bone_ids = bone_ids

    def to_dict(self):
        return {"id": self.id, "x": self.x, "y": self.y, "bone_ids": self.bone_ids}

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["x"], data["y"], data["bone_ids"])