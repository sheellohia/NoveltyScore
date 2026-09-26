# One-command entry points. Override the interpreter with: make setup PYTHON=python3.11
PYTHON ?= python3.12
VENV   := backend/.venv
PY     := $(VENV)/bin/python
PORT   ?= 5173

.PHONY: setup backend frontend test report experiments doc

setup:                 ## create the venv, install backend + dev deps and frontend deps
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q -r backend/requirements-dev.txt
	cd frontend && npm install
	@test -f backend/.env || cp backend/.env.example backend/.env

backend:               ## run the API on :8000 (downloads embedding models on first start)
	cd backend && .venv/bin/uvicorn main:app --reload --reload-exclude ".venv/*" --reload-exclude ".cache/*" --port 8000

frontend:              ## run the UI (default :5173; `make frontend PORT=5180` if taken)
	cd frontend && npm run dev -- --port $(PORT)

test:                  ## run ALL tests (no LLM calls)
	cd backend && .venv/bin/python -m pytest

report:                ## tests + HTML report + coverage into results/
	cd backend && .venv/bin/python -m pytest --html=../results/test_report.html --self-contained-html \
	  --junitxml=../results/junit.xml \
	  --cov=scoring --cov=main --cov=models --cov=data_loader --cov-report=html:../results/coverage --cov-report=term

experiments:           ## full sweep from cached judge scores -> results/
	$(PY) experiments/run_experiments.py --cache-only

doc:                   ## regenerate SOLUTION.md from results/
	$(PY) experiments/generate_doc.py
