from src.models.bone import Bone
from src.models.joint import Joint
from src.models.muscle import Muscle


class Creature:
    id: str
    joints: list[Joint]
    bones: list[Bone]
    muscles: list[Muscle]

    def __init__(self, id, joints, bones, muscles):
        self.id = id
        self.joints = joints
        self.bones = bones
        self.muscles = muscles

    def to_dict(self):
        return {
            "id": str(self.id),
            "joints": [j.to_dict() for j in self.joints],
            "bones": [b.to_dict() for b in self.bones],
            "muscles": [m.to_dict() for m in self.muscles],
        }

    @classmethod
    def from_dict(cls, data):
        joints = [Joint.from_dict(j) for j in data.get("joints", [])]
        bones = [Bone.from_dict(b) for b in data.get("bones", [])]
        muscles = [Muscle.from_dict(m) for m in data.get("muscles", [])]
        return cls(data["id"], joints, bones, muscles)
