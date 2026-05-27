param(
    [string]$PythonExe = "python"
)

& $PythonExe -m src.training.trainer_stage2 --config configs/stage2_fusion.yaml

