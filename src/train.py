"""
train.py — Fine-tune LLaMA 3.1 on gretelai/synthetic_text_to_sql
Usage: python src/train.py --max_steps 500
"""

import argparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an expert SQL assistant. Given a database schema and a question, generate the correct SQL query.

### Database Schema:
{context}

### Question:
{question}

### SQL Query:
{sql}"""


def format_prompt(example):
    return {
        "text": PROMPT_TEMPLATE.format(
            context=example["sql_context"],
            question=example["sql_prompt"],
            sql=example["sql"],
        )
    }


def load_model(model_name: str, max_seq_length: int):
    print(f"📥 Chargement du modèle {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    return model, tokenizer


def main(args):
    # Modèle
    model, tokenizer = load_model(args.model_name, args.max_seq_length)

    # Dataset
    print("📥 Chargement du dataset...")
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    dataset_test = load_dataset("gretelai/synthetic_text_to_sql", split="test")
    dataset = dataset.map(format_prompt, num_proc=2)
    dataset_test = dataset_test.map(format_prompt, num_proc=2)
    print(f"✅ Train: {len(dataset):,} | Test: {len(dataset_test):,}")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset_test,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=args.max_steps if not args.full_training else -1,
            num_train_epochs=1 if args.full_training else 1,
            learning_rate=2e-4,
            warmup_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            seed=42,
        ),
    )

    print("🚀 Début du fine-tuning...")
    stats = trainer.train()
    print(f"✅ Terminé en {stats.metrics['train_runtime']:.0f}s")
    print(f"📉 Loss finale: {stats.metrics['train_loss']:.4f}")

    # Sauvegarde
    model.save_pretrained(args.output_dir + "/lora-adapters")
    tokenizer.save_pretrained(args.output_dir + "/lora-adapters")
    print(f"💾 Modèle sauvegardé dans {args.output_dir}/lora-adapters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--full_training", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./models/llama3-sql")
    args = parser.parse_args()
    main(args)