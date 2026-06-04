# 🧠 Stratégie d'Entraînement - Projet SHARP

Cette documentation détaille les choix techniques et les hyperparamètres utilisés pour l'entraînement du modèle de détection de mains. Elle sert de base de justification pour la présentation finale.

## 1. Architecture : YOLO11 Nano (`yolo11n.pt`)
*   **Pourquoi ce choix ?** : Nous avons privilégié la variante **Nano** car elle offre le meilleur compromis entre vitesse (FPS) et précision.
*   **Impact sur le modèle** : Garantit une détection fluide en temps réel même sur des machines sans GPU puissant lors du déploiement (serving).

## 2. Paramètres de Convergence
| Paramètre | Valeur | Justification Stratégique |
| :--- | :--- | :--- |
| **EPOCHS** | 200 | Permet au modèle d'apprendre plus longtemps pour compenser l'augmentation massive de données (mosaïque, etc.). |
| **PATIENCE** | 50 | Empêche l'arrêt prématuré de l'entraînement. L'IA a plus de temps pour stabiliser sa perte (loss) après avoir vu des exemples difficiles. |
| **IMG_SIZE** | 640 | Taille standard garantissant que les doigts (objets de petite taille) conservent assez de pixels pour être distingués. |

## 3. Augmentation de Données (Robustesse & Recall)
L'augmentation permet de simuler des conditions réelles à partir d'un dataset fixe. Notre stratégie vise à booster le **Rappel (Recall)**.

| Paramètre | Valeur | Effet sur l'Intelligence du Modèle |
| :--- | :--- | :--- |
| **AUG_DEGREES** | 15.0 | **Rotation** : Simule l'inclinaison naturelle de la main et du bras. Rend l'IA capable de détecter une main même si elle arrive de biais ou de côté. |
| **AUG_HSV_V** | 0.6 | **Luminosité** : Augmente la résilience aux variations d'éclairage. Crucial pour que l'IA fonctionne aussi bien dans une salle obscure qu'en plein soleil. |
| **AUG_MOSAIC** | 0.7 | **Mosaïque** : Combine 4 images en une seule. Force l'IA à apprendre à détecter des mains de tailles différentes et dans des arrière-plans encombrés. |
| **AUG_FLIPLR** | 0.5 | **Miroir** : Double virtuellement le dataset en inversant gauche/droite. Permet de traiter indifféremment les mains gauches et droites. |
| **AUG_MIXUP** | 0.1 | **Mixup** : Superpose deux images pour créer des exemples complexes. Aide l'IA à détecter des mains même si elles sont partiellement masquées ou se chevauchent (boost Recall). |

## 4. Diagnostic & Analyse des Itérations

### V1 : Base (exp-12.pt)
*   **Observation** : mAP50 stable (89%), mais rappel (79%) perfectible.
*   **Action** : Augmentation standard.

### V2 : Recall Boost (exp-13.pt)
*   **Observation** : Précision record (95%) mais Rappel en baisse (76%). Le modèle est devenu trop prudent à cause d'une mosaïque trop agressive (1.0).
*   **Action** : Mosaïque forcée au maximum.

### V3 : Équilibre Final (Cible)
*   **Stratégie** : "Desserrer" les contraintes pour regagner du Rappel sans sacrifier la Précision.
*   **Modifications** :
    *   Mosaïque réduite de 1.0 à 0.7 pour laisser passer des exemples plus simples.
    *   Rotation réduite de 25° à 15° pour limiter les distorsions extrêmes.
    *   Introduction de **Mixup (0.1)** pour apprendre la détection en conditions de superposition.
