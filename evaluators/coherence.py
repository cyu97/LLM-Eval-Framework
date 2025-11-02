from sentence_transformers import SentenceTransformer, util

class Metric:
    def __init__(self):
        self.name = "coherence"
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.model = None

    def compute(self, prompt, reference, response):
        if not self.model or not response or not reference:
            return 0.0
        try:
            ref_emb = self.model.encode(reference, convert_to_tensor=True)
            res_emb = self.model.encode(response, convert_to_tensor=True)
            score = util.cos_sim(ref_emb, res_emb).item()
            if score < 0:
                score = 0.0
            return float(score)
        except Exception:
            return 0.0