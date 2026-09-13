import json
import sys

def send(data):
    print(json.dumps(data), flush=True)

def main():
    value = 0
    target = 24
    stopped = False

    def snapshot():
        return {
            "value": value,
            "target": target,
            "stopped": stopped,
            "success": success,
            "step_count": step_count,
            "last_message": last_message,
        }


    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]

        if op == "reset":
            if op == "reset":
                value = request["value"]
                target = request["target"]

                stopped = False
                success = False
                step_count = 0

                last_message = (
                    f"runtime reset; value={value}"
                )

                send(snapshot())
        elif op == "command":
            text = request["text"].strip()
            step_count += 1

            parts = text.split()

            if text == "get":
                last_message = (
                    f"value={value}"
                )

            elif len(parts) == 2 and parts[0] == "add":
                n = int(parts[1])
                value += n
                last_message = (
                    f"added {n}; value={value}"
                )

            elif len(parts) == 2 and parts[0] == "mul":
                n = int(parts[1])
                value *= n
                last_message = (
                    f"multiplied by {n}; value={value}"
                )

            elif text == "stop":
                stopped = True
                success = (
                    value == target
                )
                last_message = (
                    f"stopped; success={success}"
                )

            else:
                last_message = (
                    f"unknown command: {text}"
                )

            send(snapshot())

        elif op == "close":
            break

if __name__ == "__main__":
    main()