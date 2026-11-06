import random
import time

import httpx

from surety.decide.debias import average_runs
from surety.types import ChoiceAnswer, DecisionState


class SystemOneClient:
    def __init__(self, base_url: str, model: str = "kev-latest", name: str = "kev",
                 max_chars: int = 40_000, permute_endpoint: bool | None = None,
                 timeout: float = 600.0, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.max_chars = max_chars
        self.permute_endpoint = permute_endpoint
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=timeout, headers=headers)

    def _post(self, path: str, body: dict) -> httpx.Response:
        return self.http.post(f"{self.base_url}{path}", json=body)

    def _body(self, text: str, question: dict) -> dict:
        return {"model": self.model, "state": text, "questions": {"q": question}}

    @staticmethod
    def _norm(probs: dict[str, float], ids: list[str]) -> dict[str, float]:
        p = {i: float(probs.get(i, 0.0)) for i in ids}
        z = sum(p.values())
        return {i: v / z for i, v in p.items()} if z > 0 else {i: 1 / len(ids) for i in ids}

    @staticmethod
    def _tokens(data: dict) -> int:
        return int((data.get("usage") or {}).get("input_tokens", 0))

    def choice(self, state: DecisionState, n_perm: int = 1, seed: int = 0) -> ChoiceAnswer:
        t0 = time.perf_counter()
        text = state.state_text[: self.max_chars]
        truncated = len(state.state_text) > self.max_chars
        ids = [o.id for o in state.options]
        labels = {o.id: o.label for o in state.options}

        def body(order: list[str]) -> dict:
            return self._body(text, {"type": "choice", "instructions": state.instructions,
                                     "criteria": {i: labels[i] for i in order}})

        raw: list[dict[str, float]] = []
        tokens = 0
        if n_perm > 1 and self.permute_endpoint is not False:
            r = self._post("/v1/systemone/permute", {"request": body(ids), "question": "q",
                                                     "n_perm": n_perm, "seed": seed})
            if r.status_code == 404 and self.permute_endpoint is None:
                self.permute_endpoint = False
            else:
                r.raise_for_status()
                self.permute_endpoint = True
                data = r.json()
                raw = [run["probabilities"] for run in data["runs"]]
                tokens = self._tokens(data)
        if not raw:
            rng = random.Random(seed)
            for k in range(max(n_perm, 1)):
                order = list(ids)
                if k:
                    rng.shuffle(order)
                r = self._post("/v1/systemone", body(order))
                r.raise_for_status()
                data = r.json()
                raw.append(data["answers"]["q"]["probabilities"])
                tokens += self._tokens(data)

        runs = [self._norm(p, ids) for p in raw]
        return ChoiceAnswer(probs=average_runs(runs), runs=runs,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            input_tokens=tokens, truncated=truncated)

    def noul(self, state_text: str, instructions: str) -> float:
        r = self._post("/v1/systemone", self._body(
            state_text[: self.max_chars], {"type": "noul", "instructions": instructions}))
        r.raise_for_status()
        return float(r.json()["answers"]["q"]["noul"])
