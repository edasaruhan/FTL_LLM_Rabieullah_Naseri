# Part I:

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv('dataset/mental_health_tweets.csv')

# Step 2: Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

# Tokenize the dataset
tokenized_texts = df['text'].apply(lambda x: tokenizer(x, padding='max_length', truncation=True, return_tensors="pt"))
labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0 if x == 'neutral' else 2)

# Step 3: Prepare the DataLoader
train_texts, val_texts, train_labels, val_labels = train_test_split(tokenized_texts, labels, test_size=0.1)
train_dataset = torch.utils.data.TensorDataset(train_texts, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_texts, val_labels)

# Define Trainer and TrainingArguments
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Step 4: Train and evaluate the model
trainer.train()
results = trainer.evaluate()
print(results)






# Part II:

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# Step 1: Load the dataset
dataset = load_dataset('climate_fever', cache_dir='dataset')

# Step 2: Preprocess the data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_function(examples):
    return tokenizer(examples['claim'], truncation=True, padding=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Step 3: Fine-tune the model
model = GPT2LMHeadModel.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# Step 4: Train and evaluate the model
trainer.train()
results = trainer.evaluate()
print(results)
