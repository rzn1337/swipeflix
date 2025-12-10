#!/bin/bash
# Quick verification script for Milestone 2 deliverables

echo "=== Milestone 2 Deliverables Verification ==="
echo ""

echo "D1: Prompt Engineering"
[ -f experiments/prompts/strategies.py ] && echo "✅ strategies.py" || echo "❌ strategies.py"
[ -f data/eval.jsonl ] && echo "✅ eval.jsonl" || echo "❌ eval.jsonl"
[ -f experiments/prompts/prompt_report.md ] && echo "✅ prompt_report.md" || echo "❌ prompt_report.md"
[ -f experiments/prompts/evaluator.py ] && echo "✅ evaluator.py" || echo "❌ evaluator.py"
echo ""

echo "D2: RAG Pipeline"
[ -f src/ingest.py ] && echo "✅ src/ingest.py" || echo "❌ src/ingest.py"
[ -f src/swipeflix/rag/ingest.py ] && echo "✅ rag/ingest.py" || echo "❌ rag/ingest.py"
[ -f src/swipeflix/api/rag_routes.py ] && echo "✅ rag_routes.py" || echo "❌ rag_routes.py"
[ -f docs/RAG_ARCHITECTURE.md ] && echo "✅ RAG_ARCHITECTURE.md" || echo "❌ RAG_ARCHITECTURE.md"
grep -q "make rag" Makefile && echo "✅ make rag target" || echo "❌ make rag target"
echo ""

echo "D3: Guardrails"
[ -f src/swipeflix/guardrails/validators.py ] && echo "✅ validators.py" || echo "❌ validators.py"
[ -f src/swipeflix/guardrails/filters.py ] && echo "✅ filters.py" || echo "❌ filters.py"
[ -f SECURITY.md ] && echo "✅ SECURITY.md" || echo "❌ SECURITY.md"
echo ""

echo "D4: Monitoring"
[ -f src/swipeflix/monitoring/llm_metrics.py ] && echo "✅ llm_metrics.py" || echo "❌ llm_metrics.py"
[ -f monitoring/grafana/dashboards/swipeflix-llm-dashboard.json ] && echo "✅ Grafana dashboard" || echo "❌ Grafana dashboard"
[ -f scripts/generate_rag_drift_report.py ] && echo "✅ RAG drift script" || echo "❌ RAG drift script"
echo ""

echo "D5: CI/CD"
[ -f .github/workflows/ci.yml ] && echo "✅ CI workflow" || echo "❌ CI workflow"
grep -q "prompt-evaluation" .github/workflows/ci.yml && echo "✅ Prompt eval job" || echo "❌ Prompt eval job"
grep -q "canary-llm" .github/workflows/ci.yml && echo "✅ LLM canary job" || echo "❌ LLM canary job"
[ -f Dockerfile.rag ] && echo "✅ Dockerfile.rag" || echo "❌ Dockerfile.rag"
echo ""

echo "D6: Documentation"
[ -f EVALUATION.md ] && echo "✅ EVALUATION.md" || echo "❌ EVALUATION.md"
grep -q "Milestone 2" README.md && echo "✅ README updated" || echo "❌ README not updated"
grep -q "RAG" README.md && echo "✅ RAG docs in README" || echo "❌ RAG docs missing"
echo ""

echo "D7: Cloud"
[ -f src/swipeflix/cloud/aws_utils.py ] && echo "✅ AWS utils" || echo "❌ AWS utils"
grep -q "AWS\|S3\|CloudWatch" README.md && echo "✅ Cloud docs" || echo "❌ Cloud docs missing"
echo ""

echo "D8: Security"
[ -f SECURITY.md ] && echo "✅ SECURITY.md" || echo "❌ SECURITY.md"
grep -q "pip-audit" .github/workflows/ci.yml && echo "✅ pip-audit in CI" || echo "❌ pip-audit missing"
echo ""

echo "Bonus Features"
grep -q "langchain\|llama" requirements-llm.txt && echo "✅ LangChain/LlamaIndex" || echo "❌ LangChain/LlamaIndex"
grep -q "ab-test" src/swipeflix/api/rag_routes.py && echo "✅ A/B testing" || echo "❌ A/B testing missing"
echo ""

echo "=== Verification Complete ==="
echo ""
echo "📸 Next: See MILESTONE2_DELIVERABLES_CHECKLIST.md for screenshot guide"
