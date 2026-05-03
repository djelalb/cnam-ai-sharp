import os
import sys

from decouple import config


def trigger_cloud_training() -> None:
    """
    Déclenche un entraînement cloud via le SDK Ultralytics.

    :return: None
    """
    print("🚀 Initialisation de l'entraînement via le SDK Ultralytics...")
    api_key: str = config("ULTRALYTICS_API_KEY", default="")

    if not api_key:
        print("❌ Erreur : ULTRALYTICS_API_KEY introuvable dans le fichier .env")
        sys.exit(1)

    os.environ["ULTRALYTICS_API_KEY"] = api_key
    print(f"🔑 Clé API chargée : {api_key[:5]}...")

    username: str = config("ULTRALYTICS_USERNAME", default="djelal-boudji")
    dataset: str = config("ULTRALYTICS_DATASET", default="hands")
    project: str = config("ULTRALYTICS_PROJECT", default="ai-sharp")
    exp_name: str = config("ULTRALYTICS_EXP_NAME", default="exp")
    epochs: int = config("ULTRALYTICS_EPOCHS", default=100, cast=int)
    patience: int = config("ULTRALYTICS_PATIENCE", default=20, cast=int)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Erreur : le package 'ultralytics' n'est pas installé.")
        print("   Lance : pip install ultralytics")
        sys.exit(1)

    print("📦 Chargement du modèle yolo26n...")
    model = YOLO("yolo26n.pt")

    dataset_uri = f"ul://{username}/datasets/{dataset}"
    project_path = f"{username}/{project}"

    print(f"📡 Dataset   : {dataset_uri}")
    print(f"📁 Projet    : {project_path}")
    print(f"🧪 Expérience: {exp_name}")
    print(f"⚙️  Epochs    : {epochs} | Patience : {patience}")
    print("🌩️  Lancement de l'entraînement cloud...")

    model.train(
        data=dataset_uri,
        epochs=epochs,
        patience=patience,
        project=project_path,
        name=exp_name,
    )

    print("✅ Entraînement terminé !")
    print(
        f"👉 Résultats sur : https://platform.ultralytics.com/{project_path}/{exp_name}"
    )


if __name__ == "__main__":
    trigger_cloud_training()
