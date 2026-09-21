@echo off
setlocal
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=config.yaml"

python scripts/00_test_access.py --config "%CONFIG%" || exit /b 1
python -m src.download.download_cmems --config "%CONFIG%" || exit /b 1
python -m src.download.download_podaac --config "%CONFIG%" || exit /b 1
python -m src.preprocessing.harmonize --config "%CONFIG%" || exit /b 1
python scripts/01_test_harmonization.py --config "%CONFIG%" || exit /b 1
python -m src.data.prepare_target --config "%CONFIG%" || exit /b 1
python -m src.data.prepare_argo_validation --config "%CONFIG%" || exit /b 1
python scripts/02_test_targets.py --config "%CONFIG%" || exit /b 1
python scripts/03_test_splits.py --config "%CONFIG%" || exit /b 1
python -m src.models.autoencoder_pretrain --config "%CONFIG%" || exit /b 1
python scripts/04_test_embedding.py --config "%CONFIG%" || exit /b 1
python -m src.train --config "%CONFIG%" || exit /b 1
python scripts/05_evaluate_training.py --config "%CONFIG%" || exit /b 1
python scripts/06_validation_report.py --config "%CONFIG%" || exit /b 1
python scripts/07_test_dashboard.py --config "%CONFIG%" || exit /b 1
python -m pytest -q tests || exit /b 1

echo OceanEmbed pipeline completed successfully.
