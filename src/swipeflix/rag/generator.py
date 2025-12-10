"""RAG Generator component for SwipeFlix.

Combines retrieval with LLM generation for grounded responses.
"""

# Import prompt strategies
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from swipeflix.llm.gemini_client import GeminiClient, get_gemini_client
from swipeflix.rag.retriever import MovieRetriever, get_retriever

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from experiments.prompts.strategies import get_strategy

    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    logger.warning("Prompt strategies not available")


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    grounded: bool  # Whether response is grounded in sources
    confidence: str  # HIGH, MEDIUM, LOW
    latency_ms: float
    tokens_used: int
    cached: bool


class RAGGenerator:
    """RAG generator combining retrieval and generation.

    Implements the full RAG pipeline:
    1. Query understanding
    2. Document retrieval
    3. Context formatting
    4. LLM generation
    5. Response validation
    """

    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        retriever: Optional[MovieRetriever] = None,
        prompt_strategy: str = "meta_prompt",
        max_context_docs: int = 5,
    ):
        self.llm_client = llm_client or get_gemini_client()
        self.retriever = retriever or get_retriever()
        self.max_context_docs = max_context_docs

        # Load prompt strategy
        if STRATEGIES_AVAILABLE:
            self.strategy = get_strategy(prompt_strategy)
        else:
            self.strategy = None

        logger.info(f"RAGGenerator initialized with strategy: {prompt_strategy}")

    def generate(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 500,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Generate a RAG response for the given query.

        Args:
            query: User question
            top_k: Number of documents to retrieve
            temperature: LLM sampling temperature
            max_tokens: Maximum response tokens
            filters: Optional retrieval filters

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        context_docs = self.retriever.get_context_for_query(
            query,
            top_k=min(top_k, self.max_context_docs),
        )

        if not context_docs:
            logger.warning(f"No documents found for query: {query[:50]}...")
            return RAGResponse(
                query=query,
                answer="I couldn't find relevant movie information for your question. Please try a different query.",
                sources=[],
                grounded=False,
                confidence="LOW",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cached=False,
            )

        # Step 2: Format prompt with context
        if self.strategy:
            prompt = self.strategy.format_prompt(query, context=context_docs)
        else:
            prompt = self._default_prompt(query, context_docs)

        # Step 3: Generate response
        gen_result = self.llm_client.generate(
            prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        answer = gen_result["text"]
        latency_ms = (time.time() - start_time) * 1000

        # Step 4: Validate grounding
        grounded, confidence = self._assess_grounding(answer, context_docs)

        # Build sources list
        sources = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "relevance": doc["relevance_score"],
            }
            for doc in context_docs
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            grounded=grounded,
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_used=gen_result["tokens_used"],
            cached=gen_result["cached"],
        )

    def _default_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
    ) -> str:
        """Default prompt when strategies not available."""
        context_text = "\n".join(
            [
                f"- {doc['title']} ({doc['year']}): {doc['plot'][:200]}"
                for doc in context
            ]
        )

        return f"""You are a movie recommendation assistant. Answer based on the following information:

{context_text}

User Question: {query}

Provide a helpful, concise answer. Cite specific movies from the context."""

    def _assess_grounding(
        self,
        answer: str,
        context: List[Dict[str, Any]],
    ) -> tuple:
        """Assess if the response is grounded in the context.

        Returns:
            (grounded: bool, confidence: str)
        """
        answer_lower = answer.lower()

        # Check how many context items are mentioned
        mentioned_sources = 0
        for doc in context:
            if doc["title"].lower() in answer_lower:
                mentioned_sources += 1
            elif doc["director"].lower() in answer_lower:
                mentioned_sources += 0.5

        # Calculate grounding score
        if len(context) > 0:
            grounding_ratio = mentioned_sources / len(context)
        else:
            grounding_ratio = 0

        # Determine confidence
        if grounding_ratio >= 0.5:
            return True, "HIGH"
        elif grounding_ratio >= 0.25:
            return True, "MEDIUM"
        elif mentioned_sources >= 1:
            return True, "LOW"
        else:
            return False, "LOW"

    def generate_blurb(
        self,
        movie_id: str,
        tone: str = "engaging",
        max_length: int = 100,
    ) -> str:
        """Generate a short swipe-card blurb for a movie.

        Args:
            movie_id: Movie identifier
            tone: Tone of blurb (engaging, casual, professional)
            max_length: Maximum character length

        Returns:
            Short movie blurb
        """
        doc = self.retriever.retrieve_by_id(movie_id)
        if not doc:
            return "Movie not found."

        prompt = f"""Generate a {tone} swipe-card blurb for this movie (max {max_length} characters):

Title: {doc.title}
Genre: {doc.genre}
Year: {doc.year}
Plot: {doc.plot[:300]}

Write ONLY the blurb, nothing else. Keep it under {max_length} characters."""

        result = self.llm_client.generate(
            prompt,
            temperature=0.8,
            max_output_tokens=100,
        )

        # Truncate if needed
        blurb = result["text"].strip()
        if len(blurb) > max_length:
            blurb = blurb[: max_length - 3] + "..."

        return blurb

    def summarize_reviews(
        self,
        movie_id: str,
        reviews: List[str],
    ) -> Dict[str, Any]:
        """Summarize reviews and extract sentiment.

        Args:
            movie_id: Movie identifier
            reviews: List of review texts

        Returns:
            Dict with summary and sentiment label
        """
        if not reviews:
            return {"summary": "No reviews available.", "sentiment": "neutral"}

        reviews_text = "\n---\n".join(reviews[:5])  # Limit to 5 reviews

        prompt = f"""Summarize these movie reviews in 2-4 sentences and determine overall sentiment:

Reviews:
{reviews_text}

Respond in this exact format:
SUMMARY: [2-4 sentence summary]
SENTIMENT: [positive/negative/mixed/neutral]"""

        result = self.llm_client.generate(
            prompt,
            temperature=0.3,
            max_output_tokens=200,
        )

        # Parse response
        text = result["text"]
        summary = "Review summary not available."
        sentiment = "neutral"

        if "SUMMARY:" in text:
            parts = text.split("SENTIMENT:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            if len(parts) > 1:
                sentiment = parts[1].strip().lower()

        return {
            "summary": summary,
            "sentiment": sentiment,
        }

    def extract_structured_data(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """Extract structured movie data from free text.

        Args:
            text: Free-form text about a movie

        Returns:
            Structured dict with extracted fields
        """
        prompt = f"""Extract structured movie information from this text. Return valid JSON only.

Text: {text}

Extract these fields (use null if not found):
- title: string
- year: number
- director: string
- genres: list of strings
- cast: list of strings (main actors)
- runtime_minutes: number

Respond with ONLY the JSON object:"""

        result = self.llm_client.generate(
            prompt,
            temperature=0.1,
            max_output_tokens=300,
        )

        # Try to parse JSON
        import json

        try:
            # Clean up response
            response = result["text"].strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse structured data: {result['text'][:100]}")
            return {}

    def generate_rationale(
        self,
        user_id: str,
        movie_id: str,
        user_history: Optional[List[str]] = None,
    ) -> str:
        """Generate a brief rationale for why a movie matches a user.

        Args:
            user_id: User identifier
            movie_id: Recommended movie ID
            user_history: Optional list of previously liked movie IDs

        Returns:
            One-line rationale string
        """
        movie = self.retriever.retrieve_by_id(movie_id)
        if not movie:
            return "Recommended based on your preferences."

        # Get info about user's history
        history_info = ""
        if user_history:
            liked_genres = set()
            liked_directors = set()
            for hist_id in user_history[:5]:
                hist_movie = self.retriever.retrieve_by_id(hist_id)
                if hist_movie:
                    for genre in hist_movie.genre.split("|"):
                        liked_genres.add(genre.strip())
                    liked_directors.add(hist_movie.director)

            if liked_genres:
                history_info = f"User enjoys: {', '.join(list(liked_genres)[:3])}"

        prompt = f"""Generate a ONE SENTENCE rationale (max 100 chars) for recommending this movie:

Movie: {movie.title} ({movie.year})
Genre: {movie.genre}
Director: {movie.director}
{history_info}

Write ONLY the rationale, nothing else:"""

        result = self.llm_client.generate(
            prompt,
            temperature=0.7,
            max_output_tokens=50,
        )

        rationale = result["text"].strip()
        # Ensure one line
        rationale = rationale.split("\n")[0]
        if len(rationale) > 100:
            rationale = rationale[:97] + "..."

        return rationale


# Global generator instance
_generator: Optional[RAGGenerator] = None


def get_rag_generator() -> RAGGenerator:
    """Get or create the global RAG generator instance."""
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator
