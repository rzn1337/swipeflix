"""Evaluation framework for prompt experiments.

Implements:
- Quantitative metrics: ROUGE-L, Embedding Cosine Similarity
- Qualitative metrics: Human-in-the-loop rubric (Factuality, Helpfulness)
- MLflow integration for experiment tracking
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not available. ROUGE metrics disabled.")

try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    SBERT_AVAILABLE = False
    logger.warning(f"sentence_transformers not available: {e}")
    logger.warning(
        "Embedding similarity metrics will use fallback method (word overlap)."
    )

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not available. Experiment tracking disabled.")


@dataclass
class EvaluationResult:
    """Result from evaluating a single prompt."""

    strategy_name: str
    query: str
    response: str
    reference: Optional[str]

    # Quantitative metrics
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    rouge_l_f1: float = 0.0
    embedding_similarity: float = 0.0

    # Qualitative metrics (human evaluation)
    factuality_score: float = 0.0  # 1-5 scale
    helpfulness_score: float = 0.0  # 1-5 scale
    relevance_score: float = 0.0  # 1-5 scale

    # Metadata
    latency_ms: float = 0.0
    tokens_used: int = 0
    cached: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "strategy_name": self.strategy_name,
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "response_length": len(self.response),
            "rouge_l_precision": self.rouge_l_precision,
            "rouge_l_recall": self.rouge_l_recall,
            "rouge_l_f1": self.rouge_l_f1,
            "embedding_similarity": self.embedding_similarity,
            "factuality_score": self.factuality_score,
            "helpfulness_score": self.helpfulness_score,
            "relevance_score": self.relevance_score,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cached": self.cached,
            "error": self.error,
        }


@dataclass
class ExperimentSummary:
    """Summary of a prompt experiment."""

    strategy_name: str
    num_samples: int
    avg_rouge_l_f1: float
    avg_embedding_similarity: float
    avg_factuality: float
    avg_helpfulness: float
    avg_latency_ms: float
    total_tokens: int
    error_rate: float
    results: List[EvaluationResult] = field(default_factory=list)


class PromptEvaluator:
    """Evaluator for prompt experiments."""

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "swipeflix-prompt-experiments",
    ):
        self.experiment_name = experiment_name

        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        else:
            self.rouge_scorer = None

        # Initialize sentence transformer for embeddings
        self._embedding_model = None
        if SBERT_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Embedding model loaded successfully")
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.warning("Will use fallback embedding similarity method")
                self._embedding_model = None

        # Initialize MLflow
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking enabled: {mlflow_tracking_uri}")

    def compute_rouge_l(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-L scores."""
        if not self.rouge_scorer:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        scores = self.rouge_scorer.score(reference, prediction)
        rouge_l = scores["rougeL"]

        return {
            "precision": rouge_l.precision,
            "recall": rouge_l.recall,
            "f1": rouge_l.fmeasure,
        }

    def compute_embedding_similarity(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if not self._embedding_model:
            # Fallback: Simple word overlap similarity
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())

            if not pred_words or not ref_words:
                return 0.0

            intersection = pred_words & ref_words
            union = pred_words | ref_words

            # Jaccard similarity as fallback
            similarity = len(intersection) / len(union) if union else 0.0
            return float(similarity)

        try:
            embeddings = self._embedding_model.encode([prediction, reference])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Embedding computation failed: {e}, using fallback")
            # Fallback to word overlap
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            intersection = pred_words & ref_words
            union = pred_words | ref_words
            similarity = len(intersection) / len(union) if union else 0.0
            return float(similarity)

    def auto_evaluate_factuality(self, response: str, context: List[Dict]) -> float:
        """Automatically estimate factuality based on source coverage.

        Checks if claims in the response can be traced to context.
        Returns a score from 1-5.
        """
        if not context:
            return 3.0  # Neutral if no context

        # Extract key facts from context
        context_facts = set()
        for doc in context:
            context_facts.add(doc.get("title", "").lower())
            context_facts.add(doc.get("director", "").lower())
            context_facts.add(str(doc.get("year", "")))
            for genre in str(doc.get("genre", "")).split("|"):
                context_facts.add(genre.lower().strip())

        # Check response for fact coverage
        response_lower = response.lower()
        matches = sum(1 for fact in context_facts if fact and fact in response_lower)
        coverage = matches / max(len(context_facts), 1)

        # Map to 1-5 scale
        if coverage >= 0.4:
            return 5.0
        elif coverage >= 0.3:
            return 4.0
        elif coverage >= 0.2:
            return 3.0
        elif coverage >= 0.1:
            return 2.0
        return 1.0

    def auto_evaluate_helpfulness(self, response: str, query: str) -> float:
        """Automatically estimate helpfulness based on response quality.

        Checks for structure, length, and query relevance.
        Returns a score from 1-5.
        """
        score = 3.0  # Start at neutral

        # Check length (not too short, not too long)
        word_count = len(response.split())
        if 50 <= word_count <= 300:
            score += 0.5
        elif word_count < 20:
            score -= 1.0

        # Check for structure (lists, formatting)
        if any(marker in response for marker in ["1.", "•", "-", "**"]):
            score += 0.5

        # Check for citations/sources
        if any(
            marker in response
            for marker in ["[Source", "[Source", "based on", "according to"]
        ):
            score += 0.5

        # Check query term coverage
        query_terms = set(query.lower().split())
        response_lower = response.lower()
        term_coverage = sum(1 for term in query_terms if term in response_lower) / max(
            len(query_terms), 1
        )
        score += term_coverage

        return min(5.0, max(1.0, score))

    def evaluate_single(
        self,
        strategy_name: str,
        query: str,
        response: str,
        reference: Optional[str] = None,
        context: Optional[List[Dict]] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        cached: bool = False,
        error: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a single prompt response."""
        result = EvaluationResult(
            strategy_name=strategy_name,
            query=query,
            response=response,
            reference=reference,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cached=cached,
            error=error,
        )

        if error:
            return result

        # Compute ROUGE-L if reference available
        if reference:
            rouge_scores = self.compute_rouge_l(response, reference)
            result.rouge_l_precision = rouge_scores["precision"]
            result.rouge_l_recall = rouge_scores["recall"]
            result.rouge_l_f1 = rouge_scores["f1"]

            # Compute embedding similarity
            result.embedding_similarity = self.compute_embedding_similarity(
                response, reference
            )

        # Auto-evaluate qualitative metrics
        result.factuality_score = self.auto_evaluate_factuality(response, context or [])
        result.helpfulness_score = self.auto_evaluate_helpfulness(response, query)
        result.relevance_score = (
            result.factuality_score + result.helpfulness_score
        ) / 2

        return result

    def run_experiment(
        self,
        strategy,
        eval_dataset: List[Dict],
        llm_client,
        context_provider=None,
    ) -> ExperimentSummary:
        """Run evaluation experiment for a prompt strategy.

        Args:
            strategy: PromptStrategy instance
            eval_dataset: List of {query, reference} dicts
            llm_client: LLM client for generation
            context_provider: Optional function to get context for query

        Returns:
            ExperimentSummary with all results
        """
        results = []
        total_tokens = 0
        errors = 0

        logger.info(
            f"Running experiment: {strategy.name} on {len(eval_dataset)} samples"
        )

        for i, sample in enumerate(eval_dataset):
            query = sample["query"]
            reference = sample.get("reference")

            # Get context if provider available
            context = None
            if context_provider:
                context = context_provider(query)

            # Format prompt
            prompt = strategy.format_prompt(query, context=context)

            # Generate response
            gen_result = llm_client.generate(
                prompt,
                temperature=0.3,  # Lower temp for evaluation
                max_output_tokens=500,
            )

            # Evaluate
            eval_result = self.evaluate_single(
                strategy_name=strategy.name,
                query=query,
                response=gen_result["text"],
                reference=reference,
                context=context,
                latency_ms=gen_result["latency_ms"],
                tokens_used=gen_result["tokens_used"],
                cached=gen_result["cached"],
                error=gen_result.get("error"),
            )

            results.append(eval_result)
            total_tokens += gen_result["tokens_used"]

            if eval_result.error:
                errors += 1

            logger.debug(
                f"Sample {i+1}/{len(eval_dataset)}: ROUGE-L F1={eval_result.rouge_l_f1:.3f}"
            )

        # Compute summary
        valid_results = [r for r in results if not r.error]

        summary = ExperimentSummary(
            strategy_name=strategy.name,
            num_samples=len(eval_dataset),
            avg_rouge_l_f1=np.mean([r.rouge_l_f1 for r in valid_results])
            if valid_results
            else 0.0,
            avg_embedding_similarity=np.mean(
                [r.embedding_similarity for r in valid_results]
            )
            if valid_results
            else 0.0,
            avg_factuality=np.mean([r.factuality_score for r in valid_results])
            if valid_results
            else 0.0,
            avg_helpfulness=np.mean([r.helpfulness_score for r in valid_results])
            if valid_results
            else 0.0,
            avg_latency_ms=np.mean([r.latency_ms for r in valid_results])
            if valid_results
            else 0.0,
            total_tokens=total_tokens,
            error_rate=errors / len(eval_dataset) if eval_dataset else 0.0,
            results=results,
        )

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            self._log_to_mlflow(summary)

        return summary

    def _log_to_mlflow(self, summary: ExperimentSummary) -> None:
        """Log experiment results to MLflow."""
        try:
            with mlflow.start_run(
                run_name=f"{summary.strategy_name}-{int(time.time())}"
            ):
                # Log parameters
                mlflow.log_param("strategy", summary.strategy_name)
                mlflow.log_param("num_samples", summary.num_samples)

                # Log metrics
                mlflow.log_metric("avg_rouge_l_f1", summary.avg_rouge_l_f1)
                mlflow.log_metric(
                    "avg_embedding_similarity", summary.avg_embedding_similarity
                )
                mlflow.log_metric("avg_factuality", summary.avg_factuality)
                mlflow.log_metric("avg_helpfulness", summary.avg_helpfulness)
                mlflow.log_metric("avg_latency_ms", summary.avg_latency_ms)
                mlflow.log_metric("total_tokens", summary.total_tokens)
                mlflow.log_metric("error_rate", summary.error_rate)

                # Log detailed results as artifact
                results_json = [r.to_dict() for r in summary.results]
                with open("results.json", "w") as f:
                    json.dump(results_json, f, indent=2)
                mlflow.log_artifact("results.json")
                os.remove("results.json")

                logger.info(f"Logged experiment to MLflow: {summary.strategy_name}")

        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")

    def compare_strategies(
        self,
        summaries: List[ExperimentSummary],
    ) -> Dict[str, Any]:
        """Compare multiple strategy results.

        Returns ranking and insights.
        """
        comparison = {
            "strategies": [],
            "best_rouge_l": None,
            "best_embedding_sim": None,
            "best_factuality": None,
            "best_helpfulness": None,
            "fastest": None,
            "most_efficient": None,  # best score per token
        }

        for summary in summaries:
            strategy_data = {
                "name": summary.strategy_name,
                "rouge_l_f1": summary.avg_rouge_l_f1,
                "embedding_similarity": summary.avg_embedding_similarity,
                "factuality": summary.avg_factuality,
                "helpfulness": summary.avg_helpfulness,
                "latency_ms": summary.avg_latency_ms,
                "tokens": summary.total_tokens,
                "error_rate": summary.error_rate,
            }
            comparison["strategies"].append(strategy_data)

        # Find bests
        if summaries:
            comparison["best_rouge_l"] = max(
                summaries, key=lambda s: s.avg_rouge_l_f1
            ).strategy_name
            comparison["best_embedding_sim"] = max(
                summaries, key=lambda s: s.avg_embedding_similarity
            ).strategy_name
            comparison["best_factuality"] = max(
                summaries, key=lambda s: s.avg_factuality
            ).strategy_name
            comparison["best_helpfulness"] = max(
                summaries, key=lambda s: s.avg_helpfulness
            ).strategy_name
            comparison["fastest"] = min(
                summaries,
                key=lambda s: s.avg_latency_ms
                if s.avg_latency_ms > 0
                else float("inf"),
            ).strategy_name

            # Most efficient = highest combined score per token
            def efficiency(s):
                combined = s.avg_rouge_l_f1 + s.avg_factuality / 5
                return combined / max(s.total_tokens, 1) * 10000

            comparison["most_efficient"] = max(summaries, key=efficiency).strategy_name

        return comparison


def load_eval_dataset(filepath: str = "data/eval.jsonl") -> List[Dict]:
    """Load evaluation dataset from JSONL file."""
    dataset = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        logger.info(f"Loaded {len(dataset)} samples from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Evaluation dataset not found: {filepath}")
    return dataset
