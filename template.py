
import os
from pathlib import Path

project_name = "hospital_mlops"

list_of_files = [

    # =========================
    # APP LAYER (API + UI)
    # =========================

    "app/main.py",
    "app/api/__init__.py",
    "app/api/routes.py",
    "app/api/prediction.py",
    "app/api/schemas.py",

    "app/ui/app.py",
    "app/ui/components/__init__.py",

    f"src/{project_name}/__init__.py",

    # components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",

    # pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",

    # configuration
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/configuration.py",

    # entity
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",

    # utilities
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/exception/custom_exception.py",
    f"src/{project_name}/logger/logging.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    f"src/{project_name}/utils/ml_utils.py",

    f"src/{project_name}/cloud/__init__.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")



