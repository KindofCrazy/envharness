import json
import sys

def send(data):
    print(json.dumps(data), flush=True)

def main():
    value = 0
    target = 24
    stopped = False

    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]

        if op == "reset":
            value = request["value"]
            target = request["target"]
            stopped = False

        elif op == "command":
            text = request["text"].strip()
            ...

        elif op == "close":
            break