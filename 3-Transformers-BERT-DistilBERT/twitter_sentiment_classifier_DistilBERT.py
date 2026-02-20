# Import libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np

import re
import string
import time
import datetime

import random

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
seed_val = 10
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# GPU integration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print(f'We will use the GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the DistilBERT tokenizer.
print('Loading DistilBERT tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Load datasets
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# Text preprocessing 
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Achieve anonymization
    text = re.sub(r"@\S+", "X", text)
    text = re.sub(r"http\S+", "http", text)

    # Remove non-ASCII characters (fix mojibake issues)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Correct the spelling mistakes
    text = re.sub(r"\b(hapy)\b", "happy", text)
    text = re.sub(r"\b(angy)\b", "angry", text)
    text = re.sub(r"\b(luv)\b", "love", text)
    text = re.sub(r"\b(amzing)\b", "amazing", text)
    text = re.sub(r"\b(terible)\b", "terrible", text)
    text = re.sub(r"\b(excelent)\b", "excellent", text)
    text = re.sub(r"\b(performnce)\b", "performance", text)
    text = re.sub(r"\b(gud)\b", "good", text)
    text = re.sub(r"\b(vry)\b", "very", text)
    text = re.sub(r"\b(fantstic)\b", "fantastic", text)
    text = re.sub(r"\b(gr8)\b", "great", text)
    text = re.sub(r"\b(horrble)\b", "horrible", text)
    text = re.sub(r"\b(im)\b", "i am", text)
    text = re.sub(r"\b(omg)\b", "oh my god", text)
    text = re.sub(r"\b(plz)\b", "please", text)
    text = re.sub(r"\b(thx)\b", "thanks", text)

    return text

# Apply preprocessing
train_df['Processed_Text'] = train_df['Text'].apply(preprocess_text)
val_df["Processed_Text"] = val_df["Text"].apply(preprocess_text)
test_df["Processed_Text"] = test_df["Text"].apply(preprocess_text)

# train_df_subset, _ = train_test_split(
#     train_df, 
#     train_size=0.05,
#     stratify=train_df['Label'],
#     random_state=seed_val
# )

# val_df_subset, _ = train_test_split(
#     val_df, 
#     train_size=0.03,
#     stratify=val_df['Label'],
#     random_state=seed_val
# )

# Separate the data
sentences_train = train_df.Processed_Text.values
labels_train = train_df.Label.values

sentences_val = val_df.Processed_Text.values
labels_val = val_df.Label.values

sentences_test = test_df.Processed_Text.values

# Maximum sentence length
max_len_calculated = 0
sentences = np.concatenate([sentences_train, sentences_val, sentences_test])
for sentence in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len_calculated = max(max_len_calculated, len(input_ids))
    
print('Max sentence length: ', max_len_calculated)

MODEL_MAX_LEN = 128

def encode_data(sentences, labels, model_max_len):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = model_max_len,
                            padding = 'max_length',
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if labels is not None:
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    return input_ids, attention_masks

x_train_ids, x_train_masks, y_train_tensor = encode_data(sentences_train, labels_train, MODEL_MAX_LEN)
x_val_ids, x_val_masks, y_val_tensor = encode_data(sentences_val, labels_val, MODEL_MAX_LEN)
x_test_ids, x_test_masks = encode_data(sentences_test, None, MODEL_MAX_LEN)

# Dataloader
batch_size = 64

train_dataset = TensorDataset(x_train_ids, x_train_masks, y_train_tensor)
val_dataset = TensorDataset(x_val_ids, x_val_masks, y_val_tensor)

# Create the DataLoaders for our training and validation sets.
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

# For test set predictions later
test_dataset = TensorDataset(x_test_ids, x_test_masks)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Load DistilBertForSequenceClassification.
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
    dropout = 0.15,
    attention_dropout = 0.1
)
model.to(device)

optimizer = AdamW(model.parameters(), lr = 2.9609614936542493e-05, weight_decay= 0.1, eps = 1e-8)

epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

warmup_ratio = 0.05
num_warmup_steps_val = int(total_steps * warmup_ratio)

# Create the learning rate scheduler to dynamically adjust the learning rate.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps_val, num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Training Loop
training_stats = []
best_val_accuracy = 0.0
no_improve_epochs = 0
early_stop_epochs = 2

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 20 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        result = model(b_input_ids,
                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True)
        
        loss = result.loss
        logits = result.logits
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print(f"\n  Average training loss: {avg_train_loss:.2f}")
    print(f"  Training epoch took: {training_time}")

    print("\nRunning Validation...")

    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)
            
        loss = result.loss
        logits = result.logits
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)

    print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")
    print(f"  Validation took: {validation_time}")

    training_stats.append({
        'epoch': epoch_i + 1,
        'Training Loss': avg_train_loss,
        'Valid. Loss': avg_val_loss,
        'Valid. Accur.': avg_val_accuracy,
        'Training Time': training_time,
        'Validation Time': validation_time
    })
    
    # Early stopping
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), 'best_distilbert_model.pth')
        print("  New best DistilBERT model saved!")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_epochs:
            print(f"  Early stopping BERT after {early_stop_epochs} epochs without improvement.")
            break

    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current LR: {current_lr:.2e}")

print("\nTraining complete!")
print(f"Total training took {format_time(time.time()-total_t0)}")

# Load the best model weights
model.load_state_dict(torch.load('best_distilbert_model.pth'))

model.eval()
all_val_preds = []
all_val_true = []
all_val_probs = []

with torch.no_grad():
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits

        # Get probabilities for ROC curve
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_val_probs.extend(probs.cpu().numpy())

        preds = torch.argmax(logits, dim=1)
        all_val_preds.extend(preds.cpu().numpy())
        all_val_true.extend(b_labels.cpu().numpy())

val_preds_np = np.array(all_val_preds)
val_true_np = np.array(all_val_true)
val_probs_np = np.array(all_val_probs)

accuracy = accuracy_score(val_true_np, val_preds_np)
precision = precision_score(val_true_np, val_preds_np)
recall = recall_score(val_true_np, val_preds_np)
f1 = f1_score(val_true_np, val_preds_np)

print("\nFinal DistilBERT Model Evaluation Metrics (on Validation Set):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report (DistilBERT):")
print(classification_report(val_true_np, val_preds_np))

model.eval()
test_predictions = [] # Renamed
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        test_predictions.extend(preds.cpu().numpy())

test_preds_np = np.array(test_predictions).flatten().astype(int)

# Plots
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')

plt.figure(figsize=(10, 6))
plt.plot(df_stats['Valid. Accur.'], 'g-o', label="Validation Accuracy (DistilBERT)")
plt.plot(df_stats['Training Loss'], 'b-o', label="Training Loss (DistilBERT)")
plt.plot(df_stats['Valid. Loss'], 'r-o', label="Validation Loss (DistilBERT)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Loss")
plt.title("DistilBERT Training & Validation Accuracy and Loss")
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(val_true_np, val_probs_np)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DistilBERT ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - DistilBERT')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

cm = confusion_matrix(val_true_np, val_preds_np)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - DistilBERT')
plt.grid(False)
plt.tight_layout()
plt.show()

# Create submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': test_preds_np})
submission_df.to_csv("submission_distilbert.csv", index=False)
print("DistilBERT Test set predictions saved to 'submission_distilbert.csv'")