from core.model_loader import load_model
from core.dataset_loader import load_dataset
from core.evaluator import Evaluator
from core.report import Report
from tqdm import tqdm

class EvalRunner:
    def __init__(self, model_name, dataset_path):
        self.model = load_model(model_name)
        self.dataset = load_dataset(dataset_path)
        self.reporter = Report()

    def run(self, metrics):
        evaluator = Evaluator(metrics)
        all_results = {metric: 0.0 for metric in metrics}
        count = 0
        for item in tqdm(self.dataset, desc="Evaluating"):
            prompt = item.get("prompt") or item.get("input") or ""
            reference = item.get("answer") or item.get("reference") or ""
            try:
                response = self.model.generate(prompt)
            except Exception:
                response = ""
            scores = evaluator.evaluate(prompt, reference, response)
            for m, s in scores.items():
                all_results[m] += s
            count += 1
        if count == 0:
            return all_results
        for m in all_results:
            all_results[m] /= count
        return all_results

    def save_report(self, results):
        self.reporter.save(results)