from envharness.bridges.toy24.game import Toy24State, combine, reset_numbers, stop

class Toy24Env:
    def __init__(self):
        self.state = Toy24State()

    def reset(self, numbers, target=24):
        self.state = Toy24State(
            target=target,
            initial_numbers=list(numbers),
            current_numbers=[float(n) for n in numbers]
        )

        return self.observe()

    def step(self, action: dict):
        self.state.step_count += 1

        if action["name"] == "combine":
            try:
                combine(self.state, action["kwargs"]["i"], action["kwargs"]["j"], action["kwargs"]["op"])
            except (TypeError, KeyError):
                    return {
                        "observation": self.observe(),
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "info": {
                            "error": "bad args" 
                        }
                    }
        elif action["name"] == "reset":
            reset_numbers(self.state)
        elif action["name"] == "stop":
            stop(self.state)
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

        return {
            "observation": self.observe(),
            "reward": 1.0 if self.state.stopped and self.state.success else 0.0,
            "terminated": self.state.stopped,
            "truncated": False,
        }

    def observe(self):
        return {
            "text": f"target={self.state.target}, numbers={self.state.current_numbers}, history={self.state.history}, step_count={self.state.step_count}",
            "data": {
                "numbers": list(self.state.current_numbers),
                "target": self.state.target,
                "history": list(self.state.history),
                "step_count": self.state.step_count
            }
        }

    def evaluate(self):
        return {
            "success": self.state.stopped and self.state.success,
            "score": 1.0 if self.state.stopped and self.state.success else 0.0,
            "metrics": {
                "steps": self.state.step_count,
            }
        }

