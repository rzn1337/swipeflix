"""Tests for LLM components."""

import pytest


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        from swipeflix.llm.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=5,
            tokens_per_minute=1000,
            requests_per_day=20,
        )

        assert limiter.rpm_limit == 5
        assert limiter.tpm_limit == 1000
        assert limiter.daily_limit == 20

    def test_can_make_request_initially(self):
        """Test that requests are allowed initially."""
        from swipeflix.llm.rate_limiter import RateLimiter

        limiter = RateLimiter()
        assert limiter.can_make_request(100)

    def test_record_request(self):
        """Test recording a request."""
        from swipeflix.llm.rate_limiter import RateLimiter

        limiter = RateLimiter()
        limiter.record_request(100)

        status = limiter.get_status()
        assert status["requests_this_minute"] == 1
        assert status["tokens_this_minute"] == 100

    def test_rate_limit_blocking(self):
        """Test that rate limits block requests."""
        from swipeflix.llm.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=2)

        # Record max requests
        limiter.record_request(10)
        limiter.record_request(10)

        # Should be blocked
        assert not limiter.can_make_request(10)

    def test_get_remaining_quota(self):
        """Test getting remaining quota."""
        from swipeflix.llm.rate_limiter import RateLimiter

        limiter = RateLimiter(
            requests_per_minute=5,
            requests_per_day=20,
        )

        # Initial quota
        remaining = limiter.get_remaining_quota()
        assert remaining["rpm_remaining"] == 5
        assert remaining["daily_remaining"] == 20

        # After one request
        limiter.record_request(100)
        remaining = limiter.get_remaining_quota()
        assert remaining["rpm_remaining"] == 4
        assert remaining["daily_remaining"] == 19


class TestGeminiClient:
    """Tests for GeminiClient."""

    def test_client_without_api_key(self):
        """Test client behavior without API key."""
        import os

        # Temporarily remove API key
        old_key = os.environ.pop("GEMINI_API_KEY", None)

        try:
            from swipeflix.llm.gemini_client import GeminiClient

            # Create new client without key
            client = GeminiClient(api_key=None)

            # Should not be available
            assert not client.is_available()

            # Generate should return error message
            result = client.generate("Test prompt")
            assert "not available" in result["text"].lower() or "error" in result

        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        from swipeflix.llm.gemini_client import GeminiClient

        client = GeminiClient.__new__(GeminiClient)
        client._initialized = False

        key1 = client._get_cache_key("test prompt", temperature=0.7)
        key2 = client._get_cache_key("test prompt", temperature=0.7)
        key3 = client._get_cache_key("different prompt", temperature=0.7)

        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different inputs = different key

    def test_token_estimation(self):
        """Test token estimation."""
        from swipeflix.llm.gemini_client import GeminiClient

        client = GeminiClient.__new__(GeminiClient)

        # Rough estimation: ~4 chars per token
        short_text = "Hello world"  # ~11 chars
        long_text = "A" * 400  # 400 chars

        short_tokens = client._estimate_tokens(short_text)
        long_tokens = client._estimate_tokens(long_text)

        assert short_tokens < long_tokens
        assert long_tokens >= 100  # Should be ~100 tokens


class TestPromptStrategies:
    """Tests for prompt strategies."""

    def test_zero_shot_strategy(self):
        """Test zero-shot strategy formatting."""
        from experiments.prompts.strategies import ZeroShotStrategy

        strategy = ZeroShotStrategy()
        assert strategy.name == "zero_shot"

        prompt = strategy.format_prompt("What movies do you recommend?")
        assert "SwipeFlix" in prompt
        assert "recommend" in prompt.lower()

    def test_few_shot_strategy(self):
        """Test few-shot strategy formatting."""
        from experiments.prompts.strategies import FewShotStrategy

        strategy = FewShotStrategy(num_examples=3)
        assert strategy.name == "few_shot_k3"

        prompt = strategy.format_prompt("What movies do you recommend?")
        assert "Example" in prompt
        assert "recommend" in prompt.lower()

    def test_chain_of_thought_strategy(self):
        """Test chain-of-thought strategy formatting."""
        from experiments.prompts.strategies import ChainOfThoughtStrategy

        strategy = ChainOfThoughtStrategy()
        assert strategy.name == "chain_of_thought"

        prompt = strategy.format_prompt("What movies do you recommend?")
        assert "step" in prompt.lower()
        assert "Step 1" in prompt

    def test_meta_prompt_strategy(self):
        """Test meta-prompt strategy formatting."""
        from experiments.prompts.strategies import MetaPromptStrategy

        strategy = MetaPromptStrategy()
        assert strategy.name == "meta_prompt"

        prompt = strategy.format_prompt("What movies do you recommend?")
        assert "Persona" in prompt or "SYSTEM" in prompt
        assert "Rules" in prompt or "Objectives" in prompt

    def test_strategy_with_context(self):
        """Test strategies with context documents."""
        from experiments.prompts.strategies import ZeroShotStrategy

        strategy = ZeroShotStrategy()
        context = [
            {"title": "Avatar", "year": 2009, "genre": "Sci-Fi"},
            {"title": "Titanic", "year": 1997, "genre": "Drama"},
        ]

        prompt = strategy.format_prompt("What do you recommend?", context=context)
        assert "Avatar" in prompt
        assert "Titanic" in prompt

    def test_get_strategy_factory(self):
        """Test strategy factory function."""
        from experiments.prompts.strategies import get_strategy

        strategy = get_strategy("zero_shot")
        assert strategy.name == "zero_shot"

        strategy = get_strategy("few_shot_k3")
        assert "k3" in strategy.name

        with pytest.raises(ValueError):
            get_strategy("unknown_strategy")

    def test_get_all_strategies(self):
        """Test getting all strategies."""
        from experiments.prompts.strategies import get_all_strategies

        strategies = get_all_strategies()
        assert len(strategies) >= 4  # At least 4 strategies

        names = [s.name for s in strategies]
        assert "zero_shot" in names
        assert "chain_of_thought" in names


class TestPromptEvaluator:
    """Tests for PromptEvaluator."""

    def test_rouge_computation(self):
        """Test ROUGE-L computation."""
        try:
            from experiments.prompts.evaluator import PromptEvaluator

            evaluator = PromptEvaluator()

            # Identical texts should have high score
            scores = evaluator.compute_rouge_l(
                "I recommend Avatar (2009)",
                "I recommend Avatar (2009)",
            )
            assert scores["f1"] > 0.9

            # Different texts should have lower score
            scores = evaluator.compute_rouge_l(
                "I recommend Avatar",
                "Watch Titanic instead",
            )
            assert scores["f1"] < 0.5

        except ImportError:
            pytest.skip("rouge_score not available")

    def test_embedding_similarity(self):
        """Test embedding similarity computation."""
        try:
            from experiments.prompts.evaluator import PromptEvaluator

            evaluator = PromptEvaluator()

            # Similar texts should have high similarity
            sim = evaluator.compute_embedding_similarity(
                "I love action movies",
                "Action films are great",
            )
            assert sim > 0.5

            # Different topics should have lower similarity
            sim = evaluator.compute_embedding_similarity(
                "I love action movies",
                "The weather is nice today",
            )
            assert sim < 0.7

        except ImportError:
            pytest.skip("sentence_transformers not available")

    def test_auto_evaluate_factuality(self):
        """Test automatic factuality evaluation."""
        from experiments.prompts.evaluator import PromptEvaluator

        evaluator = PromptEvaluator()
        context = [
            {"title": "Avatar", "director": "James Cameron", "year": 2009},
        ]

        # Response mentioning context should score well
        score = evaluator.auto_evaluate_factuality(
            "I recommend Avatar (2009) by James Cameron.",
            context,
        )
        assert score >= 3.0

        # Response not mentioning context should score lower
        score = evaluator.auto_evaluate_factuality(
            "Try watching something else entirely.",
            context,
        )
        assert score <= 3.0

    def test_auto_evaluate_helpfulness(self):
        """Test automatic helpfulness evaluation."""
        from experiments.prompts.evaluator import PromptEvaluator

        evaluator = PromptEvaluator()

        # Well-structured response with list
        score = evaluator.auto_evaluate_helpfulness(
            "I recommend:\n1. Avatar\n2. Titanic\n3. The Avengers\n[Source: Database]",
            "What movies do you recommend?",
        )
        assert score >= 3.5

        # Very short response
        score = evaluator.auto_evaluate_helpfulness(
            "Watch Avatar.",
            "What movies do you recommend?",
        )
        assert score <= 3.5
