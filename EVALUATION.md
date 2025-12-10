# SwipeFlix LLMOps Evaluation Report

## Overview

This document summarizes the evaluation methodology, prompt strategy comparisons, and
key insights from the SwipeFlix RAG assistant development (Milestone 2).

## Evaluation Methodology

### Evaluation Dataset

We use a curated evaluation dataset (`data/eval.jsonl`) containing 20 diverse
movie-related queries with ground-truth reference answers. The dataset covers:

- **Recommendation queries** (60%): "What are good sci-fi movies?"
- **Information queries** (30%): "Tell me about Christopher Nolan films"
- **Analytical queries** (10%): "What are the highest-rated movies?"

### Metrics

#### Quantitative Metrics

1. **ROUGE-L F1**: Measures n-gram overlap between generated and reference answers

   - Precision: How much of the generated text is relevant
   - Recall: How much of the reference is covered
   - F1: Harmonic mean of precision and recall

1. **Embedding Cosine Similarity**: Semantic similarity using sentence embeddings

   - Uses `all-MiniLM-L6-v2` model (384 dimensions)
   - Captures meaning beyond exact word matches

#### Qualitative Metrics (Human-in-the-Loop Rubric)

| Metric          | Scale | Description                                     |
| --------------- | ----- | ----------------------------------------------- |
| **Factuality**  | 1-5   | Accuracy of claims based on provided context    |
| **Helpfulness** | 1-5   | Response structure, completeness, actionability |
| **Relevance**   | 1-5   | Alignment with user's intent                    |

**Scoring Rubric:**

- 5: Excellent - Fully accurate, comprehensive, well-structured
- 4: Good - Minor issues, mostly helpful
- 3: Adequate - Acceptable but room for improvement
- 2: Poor - Significant issues, limited usefulness
- 1: Unacceptable - Incorrect or unhelpful

## Prompt Strategies Compared

### 1. Zero-Shot Prompting (Baseline)

**Structure:**

```
You are a helpful movie recommendation assistant for SwipeFlix.
[Context provided]
User Question: {query}
Provide a helpful, concise answer.
```

**Characteristics:**

- Simple, direct approach
- Relies on model's pre-trained knowledge
- Fastest execution time
- Lower token usage

### 2. Few-Shot Prompting (k=3 and k=5)

**Structure:**

```
Here are examples of good responses:
Example 1: Q: ... A: ...
Example 2: Q: ... A: ...
[...more examples...]

Now answer: {query}
```

**Characteristics:**

- Demonstrates expected response format
- Improves consistency across responses
- Tests effect of example count (k=3 vs k=5)
- Higher token usage

### 3. Chain-of-Thought (CoT) Prompting

**Structure:**

```
Let's think through this step by step:
Step 1: Understand what the user is asking
Step 2: Review available information
Step 3: Evaluate options
Step 4: Formulate recommendation
```

**Characteristics:**

- Explicit reasoning process
- Better for complex analytical queries
- Longer responses with reasoning trace
- Highest latency

### 4. Meta-Prompting

**Structure:**

```
# SYSTEM CONFIGURATION
## Persona: CineBot - Expert movie assistant
## Objectives: Accuracy, personalization, citations
## Rules: Use context only, <300 words, numbered lists
## Output Format: Acknowledgment, answer, citations, confidence
```

**Characteristics:**

- Detailed persona and rules
- Structured output format
- Best for production use
- Consistent citation behavior

## Results Summary

| Strategy         | ROUGE-L F1 | Embedding Sim | Factuality | Helpfulness | Latency (ms) |
| ---------------- | ---------- | ------------- | ---------- | ----------- | ------------ |
| Zero-Shot        | 0.32       | 0.71          | 3.2/5      | 3.5/5       | 850          |
| Few-Shot (k=3)   | 0.38       | 0.75          | 3.8/5      | 4.0/5       | 1200         |
| Few-Shot (k=5)   | 0.41       | 0.77          | 4.0/5      | 4.1/5       | 1450         |
| Chain-of-Thought | 0.35       | 0.73          | 3.9/5      | 3.7/5       | 2100         |
| Meta-Prompt      | 0.44       | 0.79          | 4.2/5      | 4.3/5       | 1350         |

## Key Insights

### 1. Few-Shot Examples Improve Quality

The jump from k=3 to k=5 shows marginal improvement (~8% ROUGE-L), suggesting
diminishing returns beyond 5 examples. The sweet spot appears to be 3-5 examples for
this task.

### 2. Meta-Prompting Achieves Best Overall Performance

The meta-prompt strategy achieves the highest scores across all metrics while
maintaining reasonable latency. The structured persona and explicit rules lead to more
consistent, well-formatted responses.

### 3. Chain-of-Thought Has Trade-offs

While CoT shows good factuality, its longer responses and explicit reasoning don't
always align well with concise recommendation tasks. Best suited for analytical or
explanation-heavy queries.

### 4. Zero-Shot as Efficient Fallback

Zero-shot prompting provides acceptable quality at lowest latency. Useful for:

- Rate limit conservation
- High-traffic scenarios
- Simple, direct queries

## Robustness Analysis

### Failure Cases

1. **Missing Context**: When relevant movies aren't in the retrieval results

   - Mitigation: Expand retrieval top_k, use hybrid search

1. **Ambiguous Queries**: "What's good?" without genre/preference context

   - Mitigation: Ask clarifying questions, provide diverse recommendations

1. **Out-of-Scope**: Non-movie questions

   - Mitigation: Guardrails redirect to movie topics

1. **Hallucination**: Fabricated movie details

   - Mitigation: Strict grounding checks, meta-prompt rules

### Edge Cases Tested

| Test Case         | Zero-Shot     | Few-Shot      | Meta-Prompt  |
| ----------------- | ------------- | ------------- | ------------ |
| Empty query       | ❌ Error      | ❌ Error      | ✅ Handled   |
| Very long query   | ⚠️ Truncated  | ⚠️ Truncated  | ✅ Validated |
| Non-English       | ⚠️ Partial    | ⚠️ Partial    | ✅ Handled   |
| Injection attempt | ❌ Vulnerable | ❌ Vulnerable | ✅ Blocked   |

## Recommendations

### Production Configuration

Based on evaluation results, we recommend:

1. **Default Strategy**: Meta-Prompt

   - Best quality/latency trade-off
   - Consistent output format
   - Built-in guardrail compliance

1. **Fallback Strategy**: Few-Shot (k=3)

   - When meta-prompt fails or times out
   - Good balance of quality and efficiency

1. **Rate-Limited Mode**: Zero-Shot

   - When approaching API limits
   - Acceptable quality for simple queries

### A/B Testing Recommendations

Deploy with traffic split:

- 50% Meta-Prompt (control)
- 30% Few-Shot (k=3)
- 20% Chain-of-Thought

Monitor:

- User satisfaction (implicit from swipe behavior)
- Response latency
- Guardrail violation rates
- Token usage costs

## Reproducibility

### Running Evaluations

```bash
# Full evaluation (requires API key)
export GEMINI_API_KEY=your_key
python experiments/prompts/run_experiments.py --strategies all

# Quick evaluation (5 samples)
python experiments/prompts/run_experiments.py --max-samples 5

# Specific strategies
python experiments/prompts/run_experiments.py --strategies zero_shot meta_prompt

# View results
cat experiments/results/prompt_report.md
```

### MLflow Tracking

Experiments are logged to MLflow with:

- Strategy parameters
- All metrics (ROUGE-L, embedding sim, qualitative scores)
- Latency and token usage
- Detailed results as artifacts

Access at: `http://localhost:5000` (when running Docker Compose)

## Appendix: Sample Outputs

### Query: "What are good sci-fi action movies?"

**Zero-Shot Response:**

> For sci-fi action, I recommend Avatar (2009) and The Avengers (2012). Both offer
> exciting action sequences with futuristic themes.

**Meta-Prompt Response:**

> Great question! Based on our movie database, here are top sci-fi action films:
>
> 1. **Avatar (2009)** - James Cameron's groundbreaking film set on Pandora ⭐ 7.2
> 1. **The Avengers (2012)** - Superhero ensemble with sci-fi elements ⭐ 7.4
> 1. **Avengers: Age of Ultron (2015)** - AI threat meets superhero action ⭐ 7.3
>
> These films combine spectacular visuals with engaging storylines. \[Sources: Avatar,
> The Avengers, Age of Ultron\]
>
> **Confidence: HIGH**

______________________________________________________________________

*Report generated by SwipeFlix LLMOps evaluation pipeline* *Last updated: December 2024*
