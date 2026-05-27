param(
    [string]$PythonExe = "python"
)

& $PythonExe -m src.training.trainer_stage1 --config configs/stage1_vision.yaml

