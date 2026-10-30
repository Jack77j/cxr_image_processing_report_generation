@echo off
REM Full study-level pipeline on Windows, image pre-processing variant (denoise + CLAHE). Run from this folder:  run_all.bat
REM Edit the paths in config.py first. Stops at the first failing step.

python prepare_data.py || goto :error
python eda.py --split train || goto :error
python select_subset.py --images 25000 || goto :error
python preprocess_images.py --csv outputs\processed\train_subset_25k.csv outputs\processed\val.csv outputs\processed\test.csv --max_images 1 || goto :error
python train.py --run_name subset25k_imgproc_v1 --use_lora --train_csv outputs\processed\train_subset_25k.csv --max_images 1 || goto :error
python generate.py --run_dir outputs\checkpoints\subset25k_imgproc_v1 || goto :error
python evaluate.py --predictions outputs\results\subset25k_imgproc_v1\predictions_test.csv || goto :error
echo.
echo All steps finished. Metrics: outputs\results\subset25k_imgproc_v1\metrics.json
goto :eof

:error
echo.
echo A step failed - see the message above.
exit /b 1
