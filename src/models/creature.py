from src.models.bone import Bone
from src.models.joint import Joint
from src.models.muscle import Muscle


class Creature:
    id: str
    joints: list[Joint]
    bones: list[Bone]
    muscles: list[Muscle]
    layer_widths: list[int]
    pretrained: bool

    def __init__(self, id, joints, bones, muscles, layer_widths, pretrained):
        self.id = id
        self.joints = joints
        self.bones = bones
        self.muscles = muscles
        self.layer_widths = layer_widths
        self.pretrained = pretrained

    def to_dict(self):
        return {
            "id": str(self.id),
            "joints": [j.to_dict() for j in self.joints],
            "bones": [b.to_dict() for b in self.bones],
            "muscles": [m.to_dict() for m in self.muscles],
            "layer_widths": self.layer_widths,
            "pretrained": self.pretrained
        }

    def get_bone(self, bone_id) -> Bone | None:
        for b in self.bones:
            if b.id == bone_id:
                return b
        return None

    def get_joint(self, joint_id) -> Joint | None:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None

    def get_muscle(self, muscle_id) -> Muscle | None:
        for m in self.muscles:
            if m.id == muscle_id:
                return m
        return None

    @classmethod
    def from_dict(cls, data):
        joints = [Joint.from_dict(j) for j in data.get("joints", [])]
        bones = [Bone.from_dict(b) for b in data.get("bones", [])]
        muscles = [Muscle.from_dict(m) for m in data.get("muscles", [])]
        layer_widths = data.get("layer_widths", [])
        return cls(data["id"], joints, bones, muscles, layer_widths, data["pretrained"])
