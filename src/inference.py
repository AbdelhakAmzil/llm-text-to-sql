"""
inference.py — Générer du SQL avec le modèle fine-tuné
Usage: python src/inference.py --model_path ./models/llama3-sql/lora-adapters
"""

import argparse
import torch
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """You are an expert SQL assistant. Given a database schema and a question, generate the correct SQL query.

### Database Schema:
{context}

### Question:
{question}

### SQL Query:
"""


def load_model(model_path: str, base_model: str, max_seq_length: int):
    print(f"📥 Chargement du modèle depuis {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_sql(model, tokenizer, question: str, schema: str, max_tokens: int = 200) -> str:
    prompt = PROMPT_TEMPLATE.format(context=schema, question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def interactive_mode(model, tokenizer):
    print("\n🦙 LLaMA 3 Text-to-SQL — Mode interactif")
    print("Tapez 'exit' pour quitter\n")
    schema = input("📋 Entrez votre schéma SQL (CREATE TABLE ...):\n> ")
    while True:
        question = input("\n❓ Question: ")
        if question.lower() == "exit":
            break
        sql = generate_sql(model, tokenizer, question, schema)
        print(f"🔍 SQL généré:\n{sql}\n")


def main(args):
    model, tokenizer = load_model(args.model_path, args.base_model, args.max_seq_length)

    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        sql = generate_sql(model, tokenizer, args.question, args.schema)
        print(f"🔍 SQL généré:\n{sql}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/llama3-sql/lora-adapters")
    parser.add_argument("--base_model", type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--question", type=str, default="What is the average salary per department?")
    parser.add_argument("--schema", type=str, default="CREATE TABLE employees (id INT, name TEXT, salary FLOAT, department TEXT);")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    args = parser.parse_args()
    main(args)