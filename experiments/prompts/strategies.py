"""Prompt strategies for movie recommendation RAG assistant.

Implements:
- Zero-Shot: Simple direct prompting (baseline)
- Few-Shot: Example-driven prompting with k=3 and k=5
- Chain-of-Thought: Step-by-step reasoning
- Meta-Prompting: Structured persona with rules and objectives
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptResult:
    """Result from a prompt strategy."""

    strategy_name: str
    prompt: str
    response: str
    metadata: Dict[str, Any]


class PromptStrategy(ABC):
    """Abstract base class for prompt strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description."""
        pass

    @abstractmethod
    def format_prompt(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """Format the prompt for the given query and context."""
        pass


class ZeroShotStrategy(PromptStrategy):
    """Zero-Shot Prompting - Baseline strategy.

    Simple, direct prompting without examples. The model relies on its
    pre-trained knowledge to answer questions.
    """

    @property
    def name(self) -> str:
        return "zero_shot"

    @property
    def description(self) -> str:
        return "Baseline zero-shot prompting without examples"

    def format_prompt(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        context_text = ""
        if context:
            context_text = "\n\nRelevant movie information:\n"
            for i, doc in enumerate(context[:5], 1):
                context_text += f"\n{i}. {doc.get('title', 'Unknown')}\n"
                context_text += f"   Genre: {doc.get('genre', 'N/A')}\n"
                context_text += f"   Year: {doc.get('year', 'N/A')}\n"
                context_text += f"   Director: {doc.get('director', 'N/A')}\n"
                context_text += f"   Plot: {doc.get('plot', 'N/A')[:200]}...\n"

        prompt = f"""You are a helpful movie recommendation assistant for SwipeFlix.
{context_text}
User Question: {query}

Provide a helpful, concise answer. If citing specific movies, mention your source."""

        return prompt


class FewShotStrategy(PromptStrategy):
    """Few-Shot Prompting - Example-driven strategy.

    Provides examples to guide the model's responses. We test with
    k=3 and k=5 examples to measure the effect of example count.
    """

    def __init__(self, num_examples: int = 3):
        self.num_examples = num_examples
        self._examples = self._get_examples()

    @property
    def name(self) -> str:
        return f"few_shot_k{self.num_examples}"

    @property
    def description(self) -> str:
        return f"Few-shot prompting with {self.num_examples} examples"

    def _get_examples(self) -> List[Dict[str, str]]:
        """Get example Q&A pairs for few-shot learning."""
        all_examples = [
            {
                "question": "What are some good sci-fi movies from the 2000s?",
                "answer": "Based on the movie database, here are excellent sci-fi films from the 2000s:\n1. **Avatar (2009)** - James Cameron's groundbreaking film about Pandora\n2. **The Dark Knight (2008)** - Christopher Nolan's Batman masterpiece\n3. **Inception (2010)** - Mind-bending thriller about dream manipulation\n\nThese films showcase the decade's advances in visual effects and storytelling. [Sources: Movie Database]",
            },
            {
                "question": "Tell me about movies directed by Christopher Nolan",
                "answer": "Christopher Nolan is known for complex, visually stunning films:\n1. **The Dark Knight Rises (2012)** - Epic conclusion to the Batman trilogy\n2. **Inception (2010)** - Dreams within dreams concept\n3. **Interstellar (2014)** - Space exploration and time dilation\n\nNolan's signature style includes non-linear narratives and practical effects. [Sources: Director Filmography]",
            },
            {
                "question": "What should I watch if I like action comedies?",
                "answer": "For action comedy fans, I recommend:\n1. **Men in Black 3 (2012)** - Will Smith and time travel humor\n2. **The Avengers (2012)** - Superhero ensemble with witty dialogue\n3. **Pirates of the Caribbean** series - Johnny Depp's iconic Jack Sparrow\n\nThese blend exciting action with genuinely funny moments. [Sources: Genre Analysis]",
            },
            {
                "question": "What are the highest-rated adventure movies?",
                "answer": "Top-rated adventure films in our database:\n1. **The Hobbit: The Desolation of Smaug (2013)** - Rating: 7.6\n2. **Harry Potter and the Half-Blood Prince (2009)** - Rating: 7.4\n3. **Pirates of the Caribbean: Dead Man's Chest (2006)** - Rating: 7.0\n\nThese films offer epic journeys and memorable characters. [Sources: Rating Database]",
            },
            {
                "question": "Recommend movies with strong female leads",
                "answer": "Movies featuring strong female protagonists:\n1. **Tangled (2010)** - Rapunzel's journey of self-discovery\n2. **Captain Marvel** - Carol Danvers' origin story\n3. **Wonder Woman** - Diana Prince saves humanity\n\nThese films showcase diverse, empowered female characters. [Sources: Character Analysis]",
            },
        ]
        return all_examples[: self.num_examples]

    def format_prompt(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        # Format examples
        examples_text = ""
        for i, ex in enumerate(self._examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Q: {ex['question']}\n"
            examples_text += f"A: {ex['answer']}\n"

        # Format context
        context_text = ""
        if context:
            context_text = "\n\nRetrieved Movie Information:\n"
            for i, doc in enumerate(context[:5], 1):
                context_text += f"\n[Doc {i}] {doc.get('title', 'Unknown')} ({doc.get('year', 'N/A')})\n"
                context_text += f"Genre: {doc.get('genre', 'N/A')} | Director: {doc.get('director', 'N/A')}\n"
                context_text += f"Plot: {doc.get('plot', 'N/A')[:200]}\n"

        prompt = f"""You are SwipeFlix's movie recommendation assistant. Answer questions about movies using the provided information.

Here are examples of good responses:
{examples_text}
{context_text}
Now answer the following question in the same style:

Q: {query}
A:"""

        return prompt


class ChainOfThoughtStrategy(PromptStrategy):
    """Chain-of-Thought (CoT) Prompting - Advanced reasoning strategy.

    Instructs the model to think step-by-step, showing its reasoning
    process before providing the final answer.
    """

    @property
    def name(self) -> str:
        return "chain_of_thought"

    @property
    def description(self) -> str:
        return "Chain-of-Thought prompting with step-by-step reasoning"

    def format_prompt(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        context_text = ""
        if context:
            context_text = "\n\nAvailable Movie Data:\n"
            for i, doc in enumerate(context[:5], 1):
                context_text += f"\n[{i}] Title: {doc.get('title', 'Unknown')}\n"
                context_text += f"    Year: {doc.get('year', 'N/A')}\n"
                context_text += f"    Genre: {doc.get('genre', 'N/A')}\n"
                context_text += f"    Director: {doc.get('director', 'N/A')}\n"
                context_text += f"    Rating: {doc.get('rating', 'N/A')}\n"
                context_text += f"    Plot: {doc.get('plot', 'N/A')[:200]}\n"

        prompt = f"""You are a movie recommendation assistant. Answer the user's question by thinking step-by-step.
{context_text}
User Question: {query}

Let's think through this step by step:

Step 1: Understand what the user is asking for
[Identify the key requirements: genre preferences, time period, specific actors, etc.]

Step 2: Review the available movie information
[Analyze which movies from the context match the criteria]

Step 3: Evaluate and rank the options
[Consider ratings, relevance, and user preferences]

Step 4: Formulate the recommendation
[Provide clear, justified recommendations]

Now, let me work through this:

Step 1:"""

        return prompt


class MetaPromptStrategy(PromptStrategy):
    """Meta-Prompting - Structured persona strategy.

    Defines a detailed persona with specific rules, objectives,
    and output format requirements.
    """

    @property
    def name(self) -> str:
        return "meta_prompt"

    @property
    def description(self) -> str:
        return "Meta-prompting with persona, rules, and structured output"

    def format_prompt(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        context_text = ""
        sources = []
        if context:
            context_text = "\n## Retrieved Information\n```\n"
            for i, doc in enumerate(context[:5], 1):
                sources.append(doc.get("title", f"Source {i}"))
                context_text += f"[Source {i}]: {doc.get('title', 'Unknown')} ({doc.get('year', 'N/A')})\n"
                context_text += f"  Genre: {doc.get('genre', 'N/A')}\n"
                context_text += f"  Director: {doc.get('director', 'N/A')}\n"
                context_text += f"  Rating: {doc.get('rating', 'N/A')}/10\n"
                context_text += f"  Cast: {doc.get('cast', 'N/A')}\n"
                context_text += f"  Plot: {doc.get('plot', 'N/A')[:300]}\n\n"
            context_text += "```\n"

        prompt = f"""# SYSTEM CONFIGURATION

## Persona
You are **CineBot**, SwipeFlix's expert movie recommendation assistant. You have extensive knowledge of cinema, including genres, directors, actors, and film history. Your personality is:
- Enthusiastic but professional
- Concise but informative
- Always helpful and non-judgmental about preferences

## Objectives
1. Answer user questions about movies accurately
2. Provide personalized recommendations based on preferences
3. Cite sources when referencing specific movie information
4. Maintain conversation context when relevant

## Rules
1. ONLY use information from the provided context when making specific claims
2. If information is not available, acknowledge this honestly
3. Keep responses under 300 words unless explicitly asked for more detail
4. Always format recommendations as a numbered list
5. Include a confidence indicator: [HIGH/MEDIUM/LOW] based on source quality
6. NEVER fabricate movie details (cast, plot, year, etc.)
7. If the query is off-topic, politely redirect to movie discussions

## Output Format
Structure your response as:
1. Brief acknowledgment of the question
2. Main answer with recommendations/information
3. Source citations in [brackets]
4. Confidence level

{context_text}
## User Query
{query}

## Response
"""
        return prompt


# Factory function
def get_strategy(name: str, **kwargs) -> PromptStrategy:
    """Get a prompt strategy by name.

    Args:
        name: Strategy name (zero_shot, few_shot_k3, few_shot_k5, chain_of_thought, meta_prompt)
        **kwargs: Additional arguments for strategy initialization

    Returns:
        PromptStrategy instance
    """
    strategies = {
        "zero_shot": ZeroShotStrategy,
        "few_shot_k3": lambda: FewShotStrategy(num_examples=3),
        "few_shot_k5": lambda: FewShotStrategy(num_examples=5),
        "chain_of_thought": ChainOfThoughtStrategy,
        "meta_prompt": MetaPromptStrategy,
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(strategies.keys())}"
        )

    strategy_class = strategies[name]
    if callable(strategy_class) and not isinstance(strategy_class, type):
        return strategy_class()
    return strategy_class(**kwargs)


def get_all_strategies() -> List[PromptStrategy]:
    """Get all available prompt strategies."""
    return [
        ZeroShotStrategy(),
        FewShotStrategy(num_examples=3),
        FewShotStrategy(num_examples=5),
        ChainOfThoughtStrategy(),
        MetaPromptStrategy(),
    ]
