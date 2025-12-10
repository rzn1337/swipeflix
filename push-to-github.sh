#!/bin/bash
# Bash script to push Milestone 2 to GitHub with tag v2.0-milestone2

echo "=== SwipeFlix Milestone 2 - GitHub Push Script ==="
echo ""

# Step 1: Check current status
echo "Step 1: Checking git status..."
git status --short
echo ""

# Step 2: Add all changes
echo "Step 2: Adding all changes..."
git add .
echo "✓ Files staged"
echo ""

# Step 3: Show what will be committed
echo "Step 3: Files to be committed:"
git status --short
echo ""

# Step 4: Commit
echo "Step 4: Committing changes..."
git commit -m "feat: Add Milestone 2 - LLMOps RAG Assistant

- D1: Prompt engineering workflow with multiple strategies
- D2: RAG pipeline with FAISS ingestion and inference API
- D3: Guardrails and safety mechanisms (PII, prompt injection, toxicity)
- D4: LLM evaluation and monitoring (Prometheus, Grafana, Evidently)
- D5: CI/CD for LLMOps (prompt linting, evaluation, Docker builds)
- D6: Documentation and reports (README, EVALUATION.md, SECURITY.md)
- D7: Cloud integration (AWS S3, CloudWatch)
- D8: Security and compliance (pip-audit, SECURITY.md)
- Frontend: Swipe-first UI prototype with RAG integration
- Fixes: Rate limiting, pickle loading, Windows compatibility"

if [ $? -eq 0 ]; then
    echo "✓ Committed successfully"
else
    echo "✗ Commit failed"
    exit 1
fi
echo ""

# Step 5: Push to main
echo "Step 5: Pushing to main branch..."
git push origin main
if [ $? -eq 0 ]; then
    echo "✓ Pushed to main successfully"
else
    echo "✗ Push failed"
    exit 1
fi
echo ""

# Step 6: Create tag
echo "Step 6: Creating tag v2.0-milestone2..."
git tag -a v2.0-milestone2 -m "Milestone 2: LLMOps RAG Assistant

Complete implementation of:
- Prompt engineering and evaluation
- RAG pipeline with FAISS
- Guardrails and safety
- LLM monitoring and evaluation
- CI/CD automation
- Cloud integration
- Security compliance"

if [ $? -eq 0 ]; then
    echo "✓ Tag created successfully"
else
    echo "✗ Tag creation failed"
    exit 1
fi
echo ""

# Step 7: Push tag
echo "Step 7: Pushing tag to GitHub..."
git push origin v2.0-milestone2
if [ $? -eq 0 ]; then
    echo "✓ Tag pushed successfully"
else
    echo "✗ Tag push failed"
    exit 1
fi
echo ""

# Step 8: Summary
echo "=== Summary ==="
echo "✓ All changes committed"
echo "✓ Pushed to main branch"
echo "✓ Tag v2.0-milestone2 created and pushed"
echo ""
echo "Next steps:"
echo "1. Check GitHub Actions: https://github.com/YOUR_USERNAME/swipeflix/actions"
echo "2. Verify CI/CD workflow passes"
echo "3. Check tag: https://github.com/YOUR_USERNAME/swipeflix/tags"
echo ""
echo "Done! 🚀"
