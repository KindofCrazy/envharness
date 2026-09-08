from game import Toy24State, combine

class Toy24Env:
    def __init__(self):
        self.state = Toy24State()

    def reset(self):
        self.state = Toy24State(
            target=24,
            initial_numbers=[3, 3, 8, 8],
            current_numbers=[3.0, 3.0, 8.0, 8.0],
        )

        return self.observe()

    def step(self, action: dict):
        self.state.step_count += 1

        try:
            if action["name"] == "combine":
                    combine(self.state, action["kwargs"]["i"], action["kwargs"]["j"], action["kwargs"]["op"])
            elif action["name"] == "reset":
                self.reset()
            elif action["name"] == "stop":
                self.stop()
            else:
                return {
                    "observation": self.observe(),
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {
                        "error": "unknown_action"
                    }
                }
        except TypeError:
            return {
                "observation": self.observe(),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {
                    "error": "bad args" 
                }
            }
        

        return {
            "observation": self.observe(),
            "reward": 1.0 if self.state.stopped and self.state.success else 0.0,
            "terminated": self.state.stopped,
            "truncated": False,
        }

    def observe(self):
        return {
            "numbers": list(self.state.current_numbers),
            "target": self.state.target,
            "history": list(self.state.history),
            "step_count": self.state.step_count
        }

    def evaluate(self):
        return {
            "success": self.state.stopped and self.state.success,
            "score": 1.0 if self.state.stopped and self.state.success else 0.0,
            "metrices": {
                "steps": self.state.step_count,
            }
        }

