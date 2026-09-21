PYTHON ?= python
CONFIG ?= config.yaml

.PHONY: all access download harmonize targets splits embedding train evaluate dashboard test

all: access download harmonize targets splits embedding train evaluate dashboard test

access:
	$(PYTHON) scripts/00_test_access.py --config $(CONFIG)

download:
	$(PYTHON) -m src.download.download_cmems --config $(CONFIG)
	$(PYTHON) -m src.download.download_podaac --config $(CONFIG)

harmonize:
	$(PYTHON) -m src.preprocessing.harmonize --config $(CONFIG)
	$(PYTHON) scripts/01_test_harmonization.py --config $(CONFIG)

targets:
	$(PYTHON) -m src.data.prepare_target --config $(CONFIG)
	$(PYTHON) -m src.data.prepare_argo_validation --config $(CONFIG)
	$(PYTHON) scripts/02_test_targets.py --config $(CONFIG)

splits:
	$(PYTHON) scripts/03_test_splits.py --config $(CONFIG)

embedding:
	$(PYTHON) -m src.models.autoencoder_pretrain --config $(CONFIG)
	$(PYTHON) scripts/04_test_embedding.py --config $(CONFIG)

train:
	$(PYTHON) -m src.train --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/05_evaluate_training.py --config $(CONFIG)
	$(PYTHON) scripts/06_validation_report.py --config $(CONFIG)

dashboard:
	$(PYTHON) scripts/07_test_dashboard.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q tests
