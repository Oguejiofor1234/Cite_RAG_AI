#!/usr/bin/env bash
# ==============================================================================
# Cite_RAG_AI — Local CI Verification Script
# Runs linting, security scans, vulnerability audits, and test suites.
# ==============================================================================
set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo -e "${BLUE}${BOLD}======================================================${NC}"
echo -e "${BLUE}${BOLD}   Cite_RAG_AI — Local Continuous Integration Check   ${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# Resolve Python executable
if [[ -f "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/venv/bin/python"
    PYTEST="${ROOT_DIR}/venv/bin/pytest"
elif command -v python3 &> /dev/null; then
    PYTHON="$(command -v python3)"
    PYTEST="$(command -v pytest || true)"
else
    echo -e "${RED}[ERROR] Python 3 not found in environment.${NC}"
    exit 1
fi

echo -e "${BLUE}[INFO] Using Python:${NC} $("${PYTHON}" --version)"

# ------------------------------------------------------------------------------
# 1. Linting & Static Code Analysis
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[1/4] Running Ruff Static Analysis & Lint Check...${NC}"
if "${PYTHON}" -m ruff check .; then
    echo -e "${GREEN}✓ Ruff check passed successfully.${NC}"
else
    echo -e "${RED}✗ Ruff lint errors detected!${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Security SAST Scan (Bandit)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[2/4] Running Bandit Security Scan...${NC}"
if "${PYTHON}" -m bandit -c pyproject.toml -r app_multi.py models/ rag/ utils/; then
    echo -e "${GREEN}✓ Bandit security analysis clean (0 high/medium vulnerabilities).${NC}"
else
    echo -e "${RED}✗ Bandit security check failed!${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Dependency Vulnerability Audit (pip-audit)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[3/4] Running pip-audit Dependency Vulnerability Scan...${NC}"
if "${PYTHON}" -m pip_audit -r requirements.txt; then
    echo -e "${GREEN}✓ No known package vulnerabilities detected.${NC}"
else
    echo -e "${RED}✗ pip-audit identified vulnerable packages!${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Pytest Test Suite & Coverage Report
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[4/4] Running Pytest Suite with Coverage...${NC}"
if "${PYTHON}" -m pytest -m "not browser" --cov=models --cov=rag --cov=utils --cov-report=term-missing; then
    echo -e "${GREEN}✓ Core unit & integration test suite passed.${NC}"
else
    echo -e "${RED}✗ Pytest test suite failed!${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# Optional: Headless Browser E2E Tests
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Optional] Checking Headless Browser E2E Tests...${NC}"
if curl -s http://127.0.0.1:8501/_stcore/health | grep -q "ok"; then
    echo -e "${BLUE}[INFO] Streamlit server is active on :8501. Running browser viewport tests...${NC}"
    "${PYTHON}" -m pytest -m browser
    echo -e "${GREEN}✓ Browser responsive & theme contrast tests passed.${NC}"
else
    echo -e "${YELLOW}[SKIP] Local Streamlit server not running on port 8501. Skipping browser tests.${NC}"
    echo -e "       (Start with 'streamlit run app_multi.py' to run browser tests)"
fi

echo -e "\n${GREEN}${BOLD}======================================================${NC}"
echo -e "${GREEN}${BOLD}   ✓ ALL CI QUALITY & SECURITY CHECKS PASSED!        ${NC}"
echo -e "${GREEN}${BOLD}======================================================${NC}\n"
