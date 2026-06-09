PY      := python
CFG     := pipeline.yaml
ENVVARS := OMP_NUM_THREADS=1

.PHONY: all init generate scan validate report run demo test lint fmt clean image

all: run

## ── Pipeline commands ──────────────────────────────────────────────────────

init:
	@$(ENVVARS) $(PY) pipeline.py init $(ARGS)

generate:
	@$(ENVVARS) $(PY) pipeline.py generate $(CFG) $(ARGS)

scan:
	@$(ENVVARS) $(PY) pipeline.py scan $(CFG) $(ARGS)

validate:
	@$(ENVVARS) $(PY) pipeline.py validate $(CFG) $(ARGS)

report:
	@$(ENVVARS) $(PY) pipeline.py report $(CFG) $(ARGS)

run:
	@$(ENVVARS) $(PY) pipeline.py run $(CFG) $(ARGS)

demo:
	@$(ENVVARS) $(PY) pipeline.py demo $(ARGS)

## ── Quick presets ───────────────────────────────────────────────────────────

quick:
	@$(MAKE) init ARGS="--quick --mechanism bubble"
	@$(MAKE) run

bubble: ARGS=--mechanism bubble
bubble: run

string:
	@$(PY) pipeline.py init --mechanism string
	@$(PY) pipeline.py run $(CFG)

## ── Dev ─────────────────────────────────────────────────────────────────────

test:
	@$(ENVVARS) pytest tests/ -v

lint:
	@ruff check .

fmt:
	@ruff format .

## ── Docker ──────────────────────────────────────────────────────────────────

TAG   ?= latest
IMG   ?= multiverse-signal-lab:$(TAG)

image:
	@docker build -t $(IMG) .

run_image:
	@docker run --rm \
		-e OMP_NUM_THREADS=1 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/runs:/app/runs \
		$(IMG) run $(CFG)

## ── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	@rm -rf runs/ artifacts/ .pytest_cache .ruff_cache __pycache__
	@find . -name "*.pyc" -delete

clean_data:
	@rm -f data/T_map.npy data/mask.npy data/cl_tt.npy
