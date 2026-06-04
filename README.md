# 🖐️ SHARP (Smart Hand Automated Recognition Project)

[![CI Pipeline](https://github.com/djelalb/cnam-ai-sharp/actions/workflows/ci.yml/badge.svg)](https://github.com/djelalb/cnam-ai-sharp/actions)

## 📋 Comprendre le projet
Le projet est séparé en deux mondes distincts :
1.  **Le Monde Local (Développement)** : Pour entraîner l'IA (via le Cloud Ultralytics) et tester le code.
2.  **Le Monde Docker (Production/Serving)** : Pour livrer l'application finale. **Docker remplace Python** : il contient son propre Python, ses propres dépendances et le modèle.

---

## 🛠️ Option A : Développement Local (Entraînement)
Utilisez cette option pour lancer la pipeline ML et générer/optimiser le modèle.

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Config (Créer .env.local avec votre API KEY)

# 3. Lancer la pipeline Cloud Training
python -m src.main all
```

---

## 🚀 Option B : Déploiement Docker (Recommandé pour la Démo)
Utilisez cette option pour lancer l'interface Web finale. **Aucune installation de Python n'est requise sur votre machine**, Docker s'occupe de tout.

```bash
# Lancer l'application complète
docker-compose up --build
```
Accès : **`http://localhost:8000`**

---

## 🏗️ Principes d'Architecture
- **SOLID** : Inférence (`inference.py`) isolée du serveur (`main.py`).
- **Type-Safe** : Validation stricte via Pydantic.
- **Propre** : Docker Multi-stage pour une image légère et sécurisée.

## 🧪 Maintenance
- **Tests** : `pytest tests/`
- **Qualité** : `ruff check .`
