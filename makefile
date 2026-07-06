makefile
.PHONY: up down ingest eval serve test

up:
	docker compose up -d

down:
	docker compose down

ingest:
	@test -n "$(DOCS)" || (echo "Usage: make ingest DOCS=./data/sample" && exit 1)
	python -m src.ingestion.pipeline --docs $(DOCS)

eval:
	@test -n "$(TESTSET)" || (echo "Usage: make eval TESTSET=./data/eval.json" && exit 1)
	python -m src.evaluation.ragas_eval --testset $(TESTSET) --output ./outputs/

serve:
	streamlit run app/streamlit_app.py --server.port 8501

test:
	pytest tests/ -v --tb=short

api:
	uvicorn src.api.main:app --reload --port 8000
