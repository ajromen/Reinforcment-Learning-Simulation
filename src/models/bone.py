class Bone:
    id: str
    joint1_id: str
    joint2_id: str

    def __init__(self, id, joint1_id, joint2_id):
        self.id = id
        self.joint1_id = joint1_id
        self.joint2_id = joint2_id

    def to_dict(self):
        return {"id": self.id, "joint1_id": self.joint1_id, "joint2_id": self.joint2_id}

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["joint1_id"], data["joint2_id"])