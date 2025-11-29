class Muscle:
    id: str
    bone1_id: str
    bone2_id: str

    def __init__(self, id, bone1_id, bone2_id):
        self.id = id
        self.bone1_id = bone1_id
        self.bone2_id = bone2_id

    def to_dict(self):
        return {"id": self.id, "bone1_id": self.bone1_id, "bone2_id": self.bone2_id}

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["bone1_id"], data["bone2_id"])