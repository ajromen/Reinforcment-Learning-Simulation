from src.models.generation import Generation


class Progress:
    creature_id: str
    generations: Generation

    def to_dict(self):
        return {
            "creature_id": self.creature_id,
            "generations": [g.to_dict() for g in self.generations],
        }

    @classmethod
    def from_dict(cls, data):
        generations = [Generation.from_dict(g) for g in data.get("generations", [])]
        return cls(data["creature_id"], generations)
