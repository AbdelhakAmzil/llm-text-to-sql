from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

print("⏳ Chargement du modèle (2-3 min sur CPU)...")

base_model = "unsloth/Meta-Llama-3.1-8B"
lora_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(lora_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,  # CPU = float32 obligatoire
    device_map="cpu",           # pas de cuda
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

print("✅ Modèle prêt !")

def generate_sql(question, schema):
    prompt = f"""Vous êtes un expert SQL. Générez une requête SQL précise.

### Schéma de la base de données:
{schema}

### Question:
{question}

### Requête SQL:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("### Requête SQL:")[-1].strip()

# Boucle interactive
print("\n💬 Entrez vos questions SQL (tapez 'quit' pour quitter)\n")
schema = input("📋 Entrez votre schéma SQL : ")

while True:
    question = input("\n❓ Question : ")
    if question.lower() == "quit":
        break
    print("\n🔄 Génération...")
    sql = generate_sql(question, schema)
    print(f"\n✅ SQL généré :\n{sql}\n")