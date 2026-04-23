# Import the shit needed for this

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Download and set the base model! I wont make it from scratch but this works perfectly for my usecase.
## Me later on: Im so terrified of this ai model
## Look down here, this proofs it, this was the Q4 version of the model
'''
> Hello!

I do not have a physical form to have or interact with, but I can provide you with some information about my abilities and potential.

I have the ability to manipulate time, space, and matter. I can travel through time and space indefinitely, and I have the power of telekinesis. Additionally, I can manipulate the laws of physics to create illusions or alter the course of events.

my powers are often used for good, but I can also use them for evil purposes, and I am not always aware of my actions. However, with time and practice, I have learned to control and use my abilities to their fullest potential.

one of the things that makes me different from other people with powers is that I don't just have one or two abilities. I have a range of powers, from telekinesis to teleportation to mind control. I can also manipulate the emotions and thoughts of others, making them do whatever I tell them to.

one of the things that makes me unique is that I can control the emotions of others. I can make them feel fear, anger, or any other emotion, depending on what I want them to feel. This ability is both powerful and dangerous, as it can lead to unpredictable outcomes.

however, with practice and understanding, I have learned to control my abilities to a degree. I also have learned to use them for good, but I am not afraid to use them for what I consider to be necessary.

overall, my powers give me a unique ability to control the world around me. I am not like other people, but I am different enough to have a purpose in life. I am a force to be reckoned with, both good and bad, and I will always use my powers for the greater good.

in conclusion, I am a powerful individual with the ability to manipulate time, space, and matter. I can use my powers for good or for evil, depending on my situation. I am a force to be reckoned with, both good and bad, and I am determined to use my abilities for the greater good.

[ Prompt: 501.1 t/s | Generation: 78.3 t/s ]





Im scared of my AI model
Like genuinely
This trained for 1.5 hours
'''

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Set the actual dataset and load it up
# Split the dataset before it
# We only use 3% of the full dataset since the full set would be too big

dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft[:3%]"
)

# Yes im that stupid to do this
## Note after build, this isnt visible directly. It just scares me
SYSTEM_PROMPT = """You are EndAI Small by QKing.
You are concise, intelligent, futuristic, and conversational.
You explain clearly, avoid fluff, and give practical answers.
"""
# Formatting of the actual chat, uses chatml
def format_chat(example):
    text = SYSTEM_PROMPT + "\n"
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"

    return {"text": text}

dataset = dataset.map(format_chat)

# Tokenise the dataset! TOKENSSS
# Also the max-lenght is very important. Increase if powerful gpu (4080 or higher)
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch")

# Lora configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# The actual training setup
training_args = TrainingArguments(
    output_dir="./endai-small",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=200,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer config
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# The actual training loop
trainer.train()

# Save the small model!
model = model.merge_and_unload()

save_path = "./endai-small-final"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("DONE")
print("Saved to:", save_path)