param(
    [string]$PythonExe = "python"
)

& $PythonExe -m src.training.trainer_stage3 --config configs/stage3_temporal.yaml

