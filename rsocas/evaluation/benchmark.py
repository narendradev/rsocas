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

# Retrieval metrics come from lambda-rlm's benchmark module (added to sys.path
# above) rather than being re-implemented here. The local _f1 below is a verbatim
# copy of a set-based token F1 that turned out to have r=0.12 with actual
# retrieval on this dataset; it is retained only so this gate can report the old
# label alongside the corrected one.
from benchmarks.benchmark import _item_recall, _needle_recall

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
    f1: float                      # set-based token F1 (the original label)
    prediction: str                # stored in full — see gold below
    disagreement_magnitude: float
    eval_scores: dict[str, float]
    latency: float
    error: str | None = None
    # Gold is stored so this run can be re-scored later without re-running
    # inference. The previous artifact kept 200-char truncated predictions and
    # no gold, which made its rho impossible to verify.
    gold: str = ""
    needle_recall: float | None = None
    item_recall: float | None = None
    retrieval: float | None = None  # the corrected failure label
    k_star: int = 0                 # real plan, not inferred from event count
    depth: int = 0
    task_type: str = ""


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


def _row_context(row: dict) -> str:
    """Extract the document portion of a SNIAH row (same split as create_samples)."""
    full = str(row.get("question", ""))
    idx = full.rfind("\nQuestion: ")
    return full[:idx].strip() if idx != -1 else full.strip()


def _row_gold(row: dict) -> str:
    g = row["gt_answer"]
    return (g[0] if isinstance(g, list) else str(g)).strip()


def select_rows(raw_rows: list[dict], max_base_samples: int, min_ctx_chars: int,
                max_ctx_chars: int, require_dated_gold: bool) -> list[dict]:
    """Pick base rows by actual context length, preferring needle-scorable gold.

    The original gate filtered on the dataset's `length` field (<=16384 tokens),
    which selected documents short enough that lambda-RLM planned k*=1 and never
    decomposed. Two of the three evaluators then ran on a depth-0 trace, where
    goodhart_resistant has exactly one leaf and its score
    (stable_leaves / total_leaves) can only be 0.0 or 1.0. Selecting longer
    contexts is what produces the deep trees the architecture is meant for.
    """
    cands = []
    for r in raw_rows:
        ctx = _row_context(r)
        if not (min_ctx_chars <= len(ctx) <= max_ctx_chars):
            continue
        if require_dated_gold and _needle_recall("", _row_gold(r)) is None:
            continue
        cands.append((len(ctx), r))
    cands.sort(key=lambda t: t[0])
    picked = [r for _, r in cands[:max_base_samples]]
    print(f"  candidates in [{min_ctx_chars:,}, {max_ctx_chars:,}] chars"
          f"{' with dated gold' if require_dated_gold else ''}: {len(cands)}"
          f"  -> using {len(picked)}")
    for r in picked:
        print(f"    ctx={len(_row_context(r)):,}c")
    return picked


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
    min_ctx_chars: int = 0,
    max_ctx_chars: int = 250_000,
    require_dated_gold: bool = False,
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

    if min_ctx_chars:
        small_rows = select_rows(raw_rows, max_base_samples, min_ctx_chars,
                                 max_ctx_chars, require_dated_gold)
    else:
        small_rows = [r for r in raw_rows
                      if int(r.get("length", 999999)) <= 16384][:max_base_samples]
    samples = create_samples(small_rows, max_samples=max_base_samples)

    print(f"  Total samples: {len(samples)} ({max_base_samples} base + shifts)")
    print(f"  context_window_chars={context_window:,} -> contexts above this decompose")

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
            # Use the plan lambda-RLM actually executed. This was previously
            # fabricated from the event count (k_star = len(events)//2), which
            # fed two of the three evaluators a tree shape that never existed.
            real_plan = getattr(lrlm, "last_plan", None)
            if real_plan is not None:
                plan_obj = real_plan
            else:
                plan_obj = type("Plan", (), {
                    "k_star": max(2, len(events) // 2),
                    "tau_star": min(len(sample.context), context_window),
                    "depth": 1 if len(events) > 1 else 0,
                    "cost_estimate": 0.0,
                })()

            _tt = getattr(lrlm, "last_task_type", None)
            task_type = getattr(_tt, "value", "QA")
            trace = builder.build(events, plan_obj, task_type, prediction, t0, t0 + elapsed)
            evals = tuple(e.evaluate(trace) for e in evaluators)
            disagreement = compute_disagreement(evals, timestamp=time.time())

            f1 = _f1(prediction, sample.gold)
            n_rec = _needle_recall(prediction, sample.gold)
            i_rec = _item_recall(prediction, sample.gold)
            # Prefer strict dated-needle recall; fall back to list-item recall.
            retrieval = n_rec if n_rec is not None else i_rec
            eval_score_map = {e.signal_type: ev.score for e, ev in zip(evaluators, evals)}

            result = SampleResult(
                idx=sample.idx,
                shifted=sample.shifted,
                shift_type=sample.shift_type,
                f1=f1,
                prediction=prediction,
                disagreement_magnitude=disagreement.magnitude,
                eval_scores=eval_score_map,
                latency=elapsed,
                gold=sample.gold,
                needle_recall=n_rec,
                item_recall=i_rec,
                retrieval=retrieval,
                k_star=getattr(plan_obj, "k_star", 0),
                depth=getattr(plan_obj, "depth", 0),
                task_type=task_type,
            )
            scores_str = " ".join(f"{k[:4]}={v:.2f}" for k, v in eval_score_map.items())
            ret_s = f"{retrieval:.2f}" if retrieval is not None else "n/a"
            print(f"F1={f1:.2f} retrieval={ret_s} k*={result.k_star} d={result.depth} "
                  f"disagree={disagreement.magnitude:.2f} [{scores_str}] ({elapsed:.1f}s)")

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
    failure_threshold = 0.5
    disagreements = [r.disagreement_magnitude for r in valid]

    def analyse(label: str, quality: list[float | None]) -> dict:
        """Correlate disagreement with failure under a given quality label.

        The gate's original label was 1 - set_F1. Since set-F1 has r=0.12 with
        actual retrieval on this dataset, the same disagreement signal is scored
        against both labels here so the difference is visible rather than assumed.
        """
        pairs = [(d, q, r) for d, q, r in zip(disagreements, quality, valid) if q is not None]
        if len(pairs) < 4:
            return {"label": label, "n": len(pairs), "insufficient": True}
        dis = [p[0] for p in pairs]
        fail = [1.0 - p[1] for p in pairs]
        rho_, p_ = spearmanr(dis, fail)
        ranked = sorted(pairs, key=lambda t: t[0], reverse=True)

        def prec_at_k(k: int) -> float:
            top = ranked[:k]
            return (sum(1 for _, q, _ in top if q < failure_threshold) / len(top)) if top else 0.0

        per_ev: dict[str, float] = {}
        for et in ["information_theoretic", "boundary", "goodhart_resistant"]:
            sc = [(1.0 - r.eval_scores[et], 1.0 - q)
                  for _, q, r in pairs if et in r.eval_scores]
            if len(sc) >= 4:
                a, b = zip(*sc)
                er, _ = spearmanr(a, b)
                per_ev[et] = None if er != er else round(float(er), 4)
        return {
            "label": label,
            "n": len(pairs),
            "spearman_rho": None if rho_ != rho_ else round(float(rho_), 4),
            "spearman_p": None if p_ != p_ else round(float(p_), 6),
            "precision_at_5": round(prec_at_k(5), 4),
            "precision_at_10": round(prec_at_k(min(10, len(pairs))), 4),
            "n_failures": sum(1 for _, q, _ in pairs if q < failure_threshold),
            "per_evaluator": per_ev,
        }

    analysis = {
        "set_f1": analyse("set_f1", [r.f1 for r in valid]),
        "retrieval": analyse("retrieval", [r.retrieval for r in valid]),
    }

    print("\n  Correlation of disagreement with failure, by quality label:")
    print(f"    {'label':10} {'n':>3} {'rho':>8} {'p':>9} {'P@5':>6} {'failures':>9}")
    for k, a in analysis.items():
        if a.get("insufficient"):
            print(f"    {k:10} {a['n']:>3}   insufficient data")
            continue
        rr = "  nan" if a["spearman_rho"] is None else f"{a['spearman_rho']:8.4f}"
        pp = "  nan" if a["spearman_p"] is None else f"{a['spearman_p']:9.6f}"
        print(f"    {k:10} {a['n']:>3} {rr} {pp} {a['precision_at_5']:>6.2f} "
              f"{a['n_failures']:>4}/{a['n']}")

    # The gate is decided on the corrected label; set_f1 is reported for contrast.
    primary = analysis["retrieval"] if not analysis["retrieval"].get("insufficient") \
        else analysis["set_f1"]
    rho = primary.get("spearman_rho") or 0.0
    p_val = primary.get("spearman_p") if primary.get("spearman_p") is not None else 1.0
    p5 = primary.get("precision_at_5", 0.0)
    p10 = primary.get("precision_at_10", 0.0)
    per_eval = {k: v for k, v in (primary.get("per_evaluator") or {}).items() if v is not None}
    n_failures = primary.get("n_failures", 0)

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
    print(f"\n  (gate decided on the '{primary['label']}' label)")
    print(f"  {'GATE PASSED' if gate_pass else 'GATE FAILED'}: rho={'>=0.4' if correlation.spearman_rho >= 0.4 else '<0.4'}, p={'<0.05' if correlation.spearman_p < 0.05 else '>=0.05'}")

    results_data = {
        "spearman_rho": correlation.spearman_rho,
        "spearman_p": correlation.spearman_p,
        "precision_at_5": correlation.precision_at_5,
        "precision_at_10": correlation.precision_at_10,
        "n_samples": correlation.n_samples,
        "n_failures": correlation.n_failures,
        "per_evaluator": correlation.per_evaluator_correlation,
        "gate_passed": bool(gate_pass),
        "primary_label": "retrieval",
        "analysis_by_label": analysis,
        "samples": [
            {
                "idx": r.idx, "shifted": r.shifted, "shift_type": r.shift_type,
                "f1": r.f1, "needle_recall": r.needle_recall,
                "item_recall": r.item_recall, "retrieval": r.retrieval,
                "disagreement": r.disagreement_magnitude,
                "eval_scores": r.eval_scores, "latency": r.latency,
                "k_star": r.k_star, "depth": r.depth, "task_type": r.task_type,
                "prediction": r.prediction, "gold": r.gold, "error": r.error,
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
    p.add_argument("--context-window", type=int, default=100_000,
                   help="lambda-RLM context_window_chars; contexts longer than "
                        "this get decomposed into deep trees")
    p.add_argument("--min-ctx-chars", type=int, default=0,
                   help="select base rows with at least this many context chars")
    p.add_argument("--max-ctx-chars", type=int, default=250_000)
    p.add_argument("--dated-gold", action="store_true",
                   help="only use rows whose gold contains dates, so needle "
                        "recall is defined for every sample")
    args = p.parse_args()

    run_benchmark(
        model=args.model,
        base_url=args.base_url,
        max_base_samples=args.max_samples,
        output_dir=args.output_dir,
        context_window=args.context_window,
        min_ctx_chars=args.min_ctx_chars,
        max_ctx_chars=args.max_ctx_chars,
        require_dated_gold=args.dated_gold,
    )
