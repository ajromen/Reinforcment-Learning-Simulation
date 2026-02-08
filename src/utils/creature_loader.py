import json
import uuid
from pathlib import Path
from typing import Tuple, List

from src.models.bone import Bone
from src.models.creature import Creature
from src.models.joint import Joint
from src.models.muscle import Muscle
from src.ui.components.joint import Joint as JointComponent
from src.ui.components.muscle import Muscle as MuscleComponent
from src.ui.components.bone import Bone as BoneComponent
from src.utils.constants import SAVE_FILE_PATH


class CreatureLoader:
    @staticmethod
    def load(file_path: str) -> Creature | None:
        creature = None
        try:
            with open(file_path, "r") as f:
                creature = Creature.from_dict(json.load(f))
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{file_path}'. Check file integrity.")

        return creature

    @staticmethod
    def ui_to_creature(joints, bones, muscles, layer_widths) -> Creature | None:
        min_x = 0  # vrati na 1000 ako hoces top left
        min_y = 0
        joints_save = []
        muscle_save = []
        bone_save = []
        for joint in joints:
            if joint.pos[0] < min_x:
                min_x = joint.pos[0]
            if joint.pos[1] < min_y:
                min_y = joint.pos[1]

        for joint in joints:
            j = Joint(joint.id, joint.pos[0] - min_x, joint.pos[1] - min_y, [bone.id for bone in joint.bones])
            joints_save.append(j)
            if len(joint.bones) == 0:
                return None

        if len(joints_save) == 0:
            return None

        for muscle in muscles:
            m = Muscle(muscle.id, muscle.bone1.id, muscle.bone2.id)
            muscle_save.append(m)

        for bone in bones:
            b = Bone(bone.id, bone.joint1.id, bone.joint2.id, [muscle.id for muscle in bone.muscles])
            bone_save.append(b)

        id = uuid.uuid4()
        creature = Creature(id, joints_save, bone_save, muscle_save, layer_widths, False)
        return creature

    @staticmethod
    def creature_to_ui(creature: Creature, scale = 25) -> Tuple[List[JointComponent], List[BoneComponent], List[MuscleComponent]]:

        joint_map = {}
        joints_ui: List[JointComponent] = []
        for j in creature.joints:
            joint_ui = JointComponent(coords=(j.x*scale, j.y*scale), pos=(j.x, j.y), id=j.id)
            joints_ui.append(joint_ui)
            joint_map[j.id] = joint_ui

        bone_map = {}
        bones_ui: List[BoneComponent] = []
        for b in creature.bones:
            joint1_ui = joint_map[b.joint1_id]
            joint2_ui = joint_map[b.joint2_id]
            bone_ui = BoneComponent(b.id, joint1_ui, joint2_ui)
            bones_ui.append(bone_ui)
            bone_map[b.id] = bone_ui

        muscles_ui: List[MuscleComponent] = []
        for m in creature.muscles:
            bone1_ui = bone_map[m.bone1_id]
            bone2_ui = bone_map[m.bone2_id]
            muscle_ui = MuscleComponent(m.id, bone1_ui, bone2_ui)
            muscles_ui.append(muscle_ui)

        return joints_ui, bones_ui, muscles_ui

    @staticmethod
    def save(joints, bones, muscles, layer_widths):
        creature = CreatureLoader.ui_to_creature(joints, bones, muscles, layer_widths)
        if creature is None: return False
        return CreatureLoader.save_creature(creature)

    @staticmethod
    def save_creature(creature: Creature):
        folder_name = SAVE_FILE_PATH + str(creature.id)
        file_name = "creature.json"
        folder_path = Path(folder_name)
        file_path = folder_path / file_name

        folder_path.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(creature.to_dict(), f, indent=4)

        return True
