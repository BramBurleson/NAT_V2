from pathlib import Path

ROOT_DATASET = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
print(ROOT_DATASET)


matches = list(Path(ROOT_DATASET).rglob('onset*.m'))