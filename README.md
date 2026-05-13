# 🦙 LLaMA 3.1 — Fine-tuning Text-to-SQL

Fine-tuning de **Meta LLaMA 3.1 8B** sur le dataset [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) pour traduire des questions en langage naturel vers des requêtes SQL précises.

---

## 🎯 Objectif

Transformer une question comme :

> _"What is the average salary per department?"_

En requête SQL :

```sql
SELECT department, AVG(salary) FROM employees GROUP BY department;
```

---

## 📦 Dataset

| Propriété      | Valeur                               |
| -------------- | ------------------------------------ |
| Source         | `gretelai/synthetic_text_to_sql`     |
| Train          | 100 000 exemples                     |
| Test           | 5 850 exemples                       |
| Domaines       | 100 (finance, santé, défense...)     |
| Complexité SQL | 8 niveaux (basic → window functions) |

---

## ⚙️ Architecture

| Composant      | Détail                         |
| -------------- | ------------------------------ |
| Modèle de base | `meta-llama/Meta-Llama-3.1-8B` |
| Méthode        | QLoRA (4-bit) + LoRA adapters  |
| Framework      | Unsloth + TRL + HuggingFace    |
| GPU requis     | T4 15GB (Colab gratuit)        |

---

## 🚀 Démarrage rapide

### Option A — Google Colab (recommandé)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

1. Ouvre le notebook `notebooks/finetune_llama3_text_to_sql.ipynb` dans Colab
2. Sélectionne **Runtime → T4 GPU**
3. Exécute toutes les cellules

### Option B — En local (GPU requis)

```bash
# 1. Cloner le repo
git clone https://github.com/ton-username/llama3-text-to-sql.git
cd llama3-text-to-sql

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'entraînement (test rapide : 500 steps)
python src/train.py --max_steps 500

# 4. Entraînement complet (1 epoch = ~100k exemples)
python src/train.py --full_training

# 5. Inférence
python src/inference.py --interactive
```

---

## 📁 Structure du projet

```
llama3-text-to-sql/
├── notebooks/
│   └── finetune_llama3_text_to_sql.ipynb  # Notebook Colab complet
├── src/
│   ├── train.py       # Script d'entraînement
│   └── inference.py   # Script d'inférence
├── models/            # Modèles sauvegardés (ignorés par git)
├── data/              # Données locales (ignorées par git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⏱️ Temps d'entraînement estimé

| Configuration        | GPU              | Steps         | Durée   |
| -------------------- | ---------------- | ------------- | ------- |
| Test rapide          | T4 (Colab free)  | 500           | ~20 min |
| Entraînement moyen   | T4 (Colab free)  | 5 000         | ~3h     |
| Entraînement complet | A100 (Colab Pro) | 100k exemples | ~2h     |

---

## 💡 Exemple d'utilisation

```python
from src.inference import load_model, generate_sql

model, tokenizer = load_model("./models/llama3-sql/lora-adapters")

schema = """
CREATE TABLE employees (id INT, name TEXT, salary FLOAT, department TEXT);
CREATE TABLE departments (id INT, name TEXT, budget FLOAT);
"""

question = "List departments where the average salary exceeds 5000?"
sql = generate_sql(model, tokenizer, question, schema)
print(sql)
# SELECT d.name, AVG(e.salary) as avg_salary
# FROM employees e JOIN departments d ON e.department = d.name
# GROUP BY d.name HAVING AVG(e.salary) > 5000;
```

---

## 📄 Licence

Ce projet est sous licence MIT. Le dataset `gretelai/synthetic_text_to_sql` est sous licence Apache 2.0.
