# Deep Learning avec ANN

Ce projet implémente un réseau de neurones artificiel (ANN) pour résoudre un problème de classification binaire basé sur le jeu de données des maladies cardiaques de l'UCI.

## 📌 Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances nécessaires.

🔧 Installation des dépendances

`pip install -r requirements.txt`

Si vous rencontrez des problèmes avec TensorFlow, assurez-vous d'avoir une version compatible de Python (>=3.7) et installez TensorFlow avec :

`pip install tensorflow-cpu # Pour une installation CPU`

Ou, si vous avez une carte graphique NVIDIA :

pip install tensorflow # Version GPU

## 🚀 Exécution du Projet

Clonez ce dépôt :

`git clone <URL_DU_REPO>
cd <NOM_DU_REPO>`

Exécutez le script principal :

`python main.py`

Ou, ouvrez le notebook Jupyter pour une exécution interactive :

jupyter notebook

## 📊 Résultats

Le modèle génère des courbes de perte et de précision, que vous pouvez visualiser après l'entraînement.

📄 Structure du projet

```
├── main.py # Code principal
├── requirements.txt # Dépendances
├── README.md # Instructions
├── dataset/ # Contient les données (si nécessaire)
└── models/ # Sauvegarde du modèle entraîné
```

## 🔥 Améliorations possibles

Optimisation des hyperparamètres

Test sur d'autres jeux de données

Utilisation de techniques avancées comme Dropout et Batch Normalization

### 👨‍💻 Développé par DAMRI ABDELLAH
