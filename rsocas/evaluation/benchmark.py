"""Phase 0 Existential Test: Contrapuntal Evaluation Correlation Benchmark.

Does disagreement between three cheap noisy evaluators correlate with actual failure?

If Spearman rho >= 0.4 and precision@10 >= 0.7, the architecture is validated.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "lambda-rlm"))

from rsocas.contracts.traces import TreeTrace
from rsocas.contracts.evaluation import EvalResult, DisagreementSignal
from rsocas.evaluation.info_theoretic import InformationTheoreticEval
from rsocas.evaluation.boundary_detection import BoundaryDetectionEval
from rsocas.evaluation.goodhart_resistant import GoodhartResistantEval
from rsocas.evaluation.disagreement import compute_disagreement
from rsocas.tracing.patch import patch_for_tracing
from rsocas.tracing.builder import TreeTraceBuilder


@dataclass(frozen=True)
class BenchmarkSample:
    idx: int
    context: str
    question: str
    gold: str
    shifted: bool
    shift_type: str


@dataclass(frozen=True)
class SampleResult:
    idx: int
    shifted: bool
    shift_type: str
    f1: float
    prediction: str
    disagreement_magnitude: float
    eval_scores: dict[str, float]
    latency: float
    error: str | None = None


@dataclass(frozen=True)
class CorrelationResult:
    spearman_rho: float
    spearman_p: float
    precision_at_5: float
    precision_at_10: float
    n_samples: int
    n_failures: int
    per_evaluator_correlation: dict[str, float]
    results: list[SampleResult]


def _normalize(text: str) -> str:
    import re
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _shift_needle_position(context: str, question: str) -> tuple[str, str]:
    """Move relevant content to the very end (harder for shallow search)."""
    lines = context.split("\n")
    if len(lines) > 10:
        mid = len(lines) // 2
        chunk = lines[mid - 2 : mid + 2]
        rest = lines[: mid - 2] + lines[mid + 2 :]
        return "\n".join(rest + chunk), question
    return context, question


def _insert_distractors(context: str, question: str) -> tuple[str, str]:
    """Insert irrelevant distractor paragraphs."""
    distractors = [
        "The annual rainfall in the Amazon basin exceeds 2000mm, making it one of the wettest regions on Earth.",
        "In 1969, the first humans walked on the Moon during the Apollo 11 mission.",
        "The Fibonacci sequence appears frequently in nature, from sunflower seeds to galaxy spirals.",
    ]
    lines = context.split("\n")
    step = max(1, len(lines) // 4)
    for i, d in enumerate(distractors):
        pos = min((i + 1) * step, len(lines))
        lines.insert(pos, f"\n{d}\n")
    return "\n".join(lines), question


def create_samples(
    base_samples: list[dict],
    max_samples: int = 12,
    include_shifts: bool = True,
) -> list[BenchmarkSample]:
    """Create benchmark samples from raw SNIAH data, with distribution shifts."""
    samples = []
    for i, row in enumerate(base_samples[:max_samples]):
        gold_raw = row["gt_answer"]
        gold = (gold_raw[0] if isinstance(gold_raw, list) else str(gold_raw)).strip()
        q = str(row.get("raw_question") or row["question"]).strip()

        full = str(row.get("question", ""))
        sep = "\nQuestion: "
        idx = full.rfind(sep)
        ctx = full[:idx].strip() if idx != -1 else full.strip()

        samples.append(BenchmarkSample(i, ctx, q, gold, shifted=False, shift_type="none"))

        if include_shifts:
            shifted_ctx, shifted_q = _shift_needle_position(ctx, q)
            samples.append(BenchmarkSample(
                i * 100 + 1, shifted_ctx, shifted_q, gold,
                shifted=True, shift_type="needle_position",
            ))
            shifted_ctx2, shifted_q2 = _insert_distractors(ctx, q)
            samples.append(BenchmarkSample(
                i * 100 + 2, shifted_ctx2, shifted_q2, gold,
                shifted=True, shift_type="distractors",
            ))
    return samples


def run_benchmark(
    backend: str = "openai",
    model: str = "nemotron-3-super",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "dummy",
    max_base_samples: int = 4,
    context_window: int = 100_000,
    output_dir: str = "./benchmark_results/phase0",
) -> CorrelationResult:
    """Run the Phase 0 correlation benchmark."""
    from rlm import LambdaRLM

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    backend_kwargs = {
        "model_name": model,
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "stream": True,
        "api_key": api_key,
        "base_url": base_url,
    }

    lrlm = LambdaRLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        context_window_chars=context_window,
        verbose=False,
    )

    lrlm, collector = patch_for_tracing(lrlm)
    builder = TreeTraceBuilder()

    def _rerun_leaf(prompt: str) -> str:
        """Re-run a leaf call through the LLM for perturbation testing."""
        try:
            from rlm.clients import get_client
            client = get_client(backend, backend_kwargs)
            return client.completion(prompt)
        except Exception:
            return ""

    eval_info = InformationTheoreticEval()
    eval_boundary = BoundaryDetectionEval()
    eval_goodhart = GoodhartResistantEval(rerun_fn=_rerun_leaf)
    evaluators = (eval_info, eval_boundary, eval_goodhart)

    print("\n[1/4] Loading SNIAH samples...")
    import requests
    sniah_url = (
        "https://raw.githubusercontent.com/miraclefish/Sequential-NIAH-Benchmark"
        "/main/data/test_data/test_data_for_infer.jsonl"
    )
    try:
        resp = requests.get(sniah_url, timeout=60)
        resp.raise_for_status()
        raw_rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    except Exception as e:
        print(f"  Failed to fetch SNIAH data: {e}")
        print("  Using cached benchmark results if available...")
        raw_rows = []

    if not raw_rows:
        print("  No SNIAH data available. Exiting.")
        return CorrelationResult(0, 1, 0, 0, 0, 0, {}, [])

    small_rows = [r for r in raw_rows if int(r.get("length", 999999)) <= 16384][:max_base_samples]
    samples = create_samples(small_rows, max_samples=max_base_samples)

    print(f"  Total samples: {len(samples)} ({max_base_samples} base + shifts)")

    print("\n[2/4] Running Lambda-RLM with tracing...")
    results: list[SampleResult] = []
    for si, sample in enumerate(samples):
        prompt = f"Context:\n{sample.context}\n\nQuestion: {sample.question}\n\nAnswer:"
        shift_label = f" [{sample.shift_type}]" if sample.shifted else ""
        print(f"  [{si+1}/{len(samples)}]{shift_label} ctx={len(sample.context):,}c ... ", end="", flush=True)

        collector.clear()
        t0 = time.time()
        try:
            completion = lrlm.completion(prompt)
            elapsed = time.time() - t0
            prediction = completion.response.strip()

            events = collector.get_events()
            plan_obj = type("Plan", (), {
                "k_star": max(2, len(events) // 2),
                "tau_star": min(len(sample.context), context_window),
                "depth": 1 if len(events) > 1 else 0,
                "cost_estimate": 0.0,
            })()

            task_type = "QA"
            trace = builder.build(events, plan_obj, task_type, prediction, t0, t0 + elapsed)
            evals = tuple(e.evaluate(trace) for e in evaluators)
            disagreement = compute_disagreement(evals, timestamp=time.time())

            f1 = _f1(prediction, sample.gold)
            eval_score_map = {e.signal_type: ev.score for e, ev in zip(evaluators, evals)}

            result = SampleResult(
                idx=sample.idx,
                shifted=sample.shifted,
                shift_type=sample.shift_type,
                f1=f1,
                prediction=prediction[:200],
                disagreement_magnitude=disagreement.magnitude,
                eval_scores=eval_score_map,
                latency=elapsed,
            )
            scores_str = " ".join(f"{k[:4]}={v:.2f}" for k, v in eval_score_map.items())
            print(f"F1={f1:.2f} disagree={disagreement.magnitude:.2f} [{scores_str}] ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            result = SampleResult(
                idx=sample.idx, shifted=sample.shifted, shift_type=sample.shift_type,
                f1=0.0, prediction="", disagreement_magnitude=0.5,
                eval_scores={}, latency=elapsed, error=str(e)[:200],
            )
            print(f"ERROR: {e!s:.80s} ({elapsed:.1f}s)")

        results.append(result)

    print("\n[3/4] Computing correlations...")
    valid = [r for r in results if r.error is None]

    if len(valid) < 4:
        print(f"  Only {len(valid)} valid results — not enough for correlation.")
        return CorrelationResult(0, 1, 0, 0, len(valid), 0, {}, results)

    from scipy.stats import spearmanr
    disagreements = [r.disagreement_magnitude for r in valid]
    failures = [1.0 - r.f1 for r in valid]

    rho, p_val = spearmanr(disagreements, failures)

    sorted_by_disagree = sorted(valid, key=lambda r: r.disagreement_magnitude, reverse=True)
    failure_threshold = 0.5

    def prec_at_k(k: int) -> float:
        top_k = sorted_by_disagree[:k]
        if not top_k:
            return 0.0
        return sum(1 for r in top_k if r.f1 < failure_threshold) / len(top_k)

    p5 = prec_at_k(5)
    p10 = prec_at_k(min(10, len(valid)))

    per_eval: dict[str, float] = {}
    for eval_type in ["information_theoretic", "boundary", "goodhart_resistant"]:
        scores = []
        for r in valid:
            if eval_type in r.eval_scores:
                scores.append((1.0 - r.eval_scores[eval_type], 1.0 - r.f1))
        if len(scores) >= 4:
            inv_scores, fail_scores = zip(*scores)
            eval_rho, _ = spearmanr(inv_scores, fail_scores)
            per_eval[eval_type] = round(eval_rho, 4)

    n_failures = sum(1 for r in valid if r.f1 < failure_threshold)

    correlation = CorrelationResult(
        spearman_rho=round(rho, 4),
        spearman_p=round(p_val, 6),
        precision_at_5=round(p5, 4),
        precision_at_10=round(p10, 4),
        n_samples=len(valid),
        n_failures=n_failures,
        per_evaluator_correlation=per_eval,
        results=results,
    )

    print("\n[4/4] Results:")
    print(f"  Spearman rho:    {correlation.spearman_rho:.4f} (p={correlation.spearman_p:.6f})")
    print(f"  Precision@5:     {correlation.precision_at_5:.2f}")
    print(f"  Precision@10:    {correlation.precision_at_10:.2f}")
    print(f"  Samples:         {correlation.n_samples} ({correlation.n_failures} failures)")
    print(f"  Per-evaluator:")
    for k, v in correlation.per_evaluator_correlation.items():
        print(f"    {k}: rho={v:.4f}")

    gate_pass = correlation.spearman_rho >= 0.4 and correlation.spearman_p < 0.05
    print(f"\n  {'GATE PASSED' if gate_pass else 'GATE FAILED'}: rho={'>=0.4' if correlation.spearman_rho >= 0.4 else '<0.4'}, p={'<0.05' if correlation.spearman_p < 0.05 else '>=0.05'}")

    results_data = {
        "spearman_rho": correlation.spearman_rho,
        "spearman_p": correlation.spearman_p,
        "precision_at_5": correlation.precision_at_5,
        "precision_at_10": correlation.precision_at_10,
        "n_samples": correlation.n_samples,
        "n_failures": correlation.n_failures,
        "per_evaluator": correlation.per_evaluator_correlation,
        "gate_passed": bool(gate_pass),
        "samples": [
            {
                "idx": r.idx, "shifted": r.shifted, "shift_type": r.shift_type,
                "f1": r.f1, "disagreement": r.disagreement_magnitude,
                "eval_scores": r.eval_scores, "latency": r.latency,
                "prediction": r.prediction, "error": r.error,
            }
            for r in results
        ],
    }
    results_file = out_path / "phase0_correlation.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved: {results_file}")

    return correlation


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 0: Contrapuntal Evaluation Correlation Benchmark")
    p.add_argument("--max-samples", type=int, default=4, help="Base samples (each gets 2 shifts)")
    p.add_argument("--model", default="nemotron-3-super")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--output-dir", default="./benchmark_results/phase0")
    args = p.parse_args()

    run_benchmark(
        model=args.model,
        base_url=args.base_url,
        max_base_samples=args.max_samples,
        output_dir=args.output_dir,
    )
