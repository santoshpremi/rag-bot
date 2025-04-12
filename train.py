from datasets import load_dataset # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model # type: ignore

# 1. Load dataset
dataset = load_dataset("squad", split="train[:500]")

# 2. Corrected data formatting
def format_data(examples):
    inputs = []
    for q, c, ans_list in zip(examples['question'], 
                            examples['context'], 
                            examples['answers']):
        # Take first answer text from answer list
        answer = ans_list['text'][0] if ans_list['text'] else ""
        inputs.append(f"Question: {q}\nContext: {c}\nAnswer: {answer}")
    return {"text": inputs}

dataset = dataset.map(format_data, batched=True)

# 3. Initialize model with LoRA
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_steps=10,
    save_strategy="no",
    report_to="none" # Disable wandb integration

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 6. Train
trainer.train()
model.save_pretrained("./fine-tuned-model")