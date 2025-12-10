# PowerShell script to push Milestone 2 to GitHub with tag v2.0-milestone2

Write-Host "=== SwipeFlix Milestone 2 - GitHub Push Script ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check current status
Write-Host "Step 1: Checking git status..." -ForegroundColor Yellow
git status --short
Write-Host ""

# Step 2: Add all changes
Write-Host "Step 2: Adding all changes..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files staged" -ForegroundColor Green
Write-Host ""

# Step 3: Show what will be committed
Write-Host "Step 3: Files to be committed:" -ForegroundColor Yellow
git status --short
Write-Host ""

# Step 4: Commit
Write-Host "Step 4: Committing changes..." -ForegroundColor Yellow
$commitMessage = @"
feat: Add Milestone 2 - LLMOps RAG Assistant

- D1: Prompt engineering workflow with multiple strategies
- D2: RAG pipeline with FAISS ingestion and inference API
- D3: Guardrails and safety mechanisms (PII, prompt injection, toxicity)
- D4: LLM evaluation and monitoring (Prometheus, Grafana, Evidently)
- D5: CI/CD for LLMOps (prompt linting, evaluation, Docker builds)
- D6: Documentation and reports (README, EVALUATION.md, SECURITY.md)
- D7: Cloud integration (AWS S3, CloudWatch)
- D8: Security and compliance (pip-audit, SECURITY.md)
- Frontend: Swipe-first UI prototype with RAG integration
- Fixes: Rate limiting, pickle loading, Windows compatibility
"@

git commit -m $commitMessage
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Committed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Commit failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Push to main
Write-Host "Step 5: Pushing to main branch..." -ForegroundColor Yellow
git push origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Pushed to main successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Push failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 6: Create tag
Write-Host "Step 6: Creating tag v2.0-milestone2..." -ForegroundColor Yellow
$tagMessage = @"
Milestone 2: LLMOps RAG Assistant

Complete implementation of:
- Prompt engineering and evaluation
- RAG pipeline with FAISS
- Guardrails and safety
- LLM monitoring and evaluation
- CI/CD automation
- Cloud integration
- Security compliance
"@

git tag -a v2.0-milestone2 -m $tagMessage
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Tag created successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Tag creation failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 7: Push tag
Write-Host "Step 7: Pushing tag to GitHub..." -ForegroundColor Yellow
git push origin v2.0-milestone2
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Tag pushed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Tag push failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 8: Summary
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "✓ All changes committed" -ForegroundColor Green
Write-Host "✓ Pushed to main branch" -ForegroundColor Green
Write-Host "✓ Tag v2.0-milestone2 created and pushed" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Check GitHub Actions: https://github.com/YOUR_USERNAME/swipeflix/actions" -ForegroundColor White
Write-Host "2. Verify CI/CD workflow passes" -ForegroundColor White
Write-Host "3. Check tag: https://github.com/YOUR_USERNAME/swipeflix/tags" -ForegroundColor White
Write-Host ""
Write-Host "Done! 🚀" -ForegroundColor Green
