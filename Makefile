PY ?= python

.PHONY: setup test smoke clean

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

# End-to-end pipeline on SYNTHETIC data only (CPU, a few minutes).
smoke:
	bash scripts/run_smoke.sh

clean:
	rm -rf runs outputs submission bundle .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
