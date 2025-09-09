CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

_metrics = {}


class Counter:
    def __init__(self, name: str, documentation: str):
        self.name = name
        self.documentation = documentation
        self.value = 0
        _metrics[name] = self

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def _render(self) -> str:
        return (
            f"# HELP {self.name} {self.documentation}\n"
            f"# TYPE {self.name} counter\n"
            f"{self.name} {self.value}\n"
        )


class Histogram:
    def __init__(self, name: str, documentation: str):
        self.name = name
        self.documentation = documentation
        self.samples = []
        _metrics[name] = self

    def observe(self, value: float) -> None:
        self.samples.append(value)

    def _render(self) -> str:
        count = len(self.samples)
        total = sum(self.samples)
        return (
            f"# HELP {self.name} {self.documentation}\n"
            f"# TYPE {self.name} histogram\n"
            f"{self.name}_count {count}\n"
            f"{self.name}_sum {total}\n"
        )


def generate_latest() -> bytes:
    out = "".join(metric._render() for metric in _metrics.values())
    return out.encode("utf-8")
