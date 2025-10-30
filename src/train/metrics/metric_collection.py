import src.train.metrics.metrics as mtrcs


METRICS = {
    "loss": mtrcs.Loss,
    "accuracy": mtrcs.Accuracy
}


class MetricCollection():
    def __init__(self, metrics: list) -> None:
        self._metrics = { name: METRICS[name]() for name in metrics }
    
    def update(self, data: dict) -> None:
        for name, kwargs in data.items():
            self._metrics[name].update(**kwargs)

    def clear(self) -> None:
        for metric in self._metrics.values():
            metric.clear()
    
    def clear_history(self) -> None:
        for metric in self._metrics.values():
            metric.clear_history()
    
    def get_history(self) -> list:
        history = { name: metric.get_history() for name, metric in self._metrics.items() }
        return history
    
    def compute(self) -> dict:
        results = { name: metric.compute() for name, metric in self._metrics.items() }
        return results
