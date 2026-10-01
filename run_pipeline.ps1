$ErrorActionPreference = 'Stop'
Write-Host "Starting Dataset Generation (5000 pairs)..."
cd src\data
& "C:\Users\HP\miniconda3\envs\gw_project\python.exe" generate_datasets.py
cd ..\..
Write-Host "Dataset Generation Complete."

Write-Host "Starting Training..."
& "C:\Users\HP\miniconda3\envs\gw_project\python.exe" src\train.py
Write-Host "Training Complete."

Write-Host "Starting Evaluation..."
& "C:\Users\HP\miniconda3\envs\gw_project\python.exe" src\evaluate.py
Write-Host "Evaluation Complete."
