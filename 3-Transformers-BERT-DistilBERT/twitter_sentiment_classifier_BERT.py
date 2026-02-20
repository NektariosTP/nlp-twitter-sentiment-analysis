# Import libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

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

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

# 1) Class Distribution (Percentage Pie Charts for Train and Validation Sets)
train_counts = train_df['Label'].value_counts(normalize=True).sort_index() * 100
val_counts = val_df['Label'].value_counts(normalize=True).sort_index() * 100
labels = ['Negative (0)', 'Positive (1)']
colors = ['tomato', 'skyblue']

plt.figure(figsize=(12, 6))

# Train set
plt.subplot(1, 2, 1)
plt.pie(train_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, counterclock=False)
plt.title('Train Set Class Distribution')
# Validation set
plt.subplot(1, 2, 2)
plt.pie(val_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, counterclock=False)
plt.title('Validation Set Class Distribution')

plt.tight_layout()
plt.show()

# 2) Average Word and Character Count per Tweet - Unprocessed and Processed
print("\nGenerating plot for average word and character counts...") # Optional: for console feedback
word_counts_unprocessed = train_df['Text'].str.split().str.len()
char_counts_unprocessed = train_df['Text'].str.len()
avg_word_unprocessed = word_counts_unprocessed.mean()
avg_char_unprocessed = char_counts_unprocessed.mean()
# Ensure you use the correct column name for processed text
word_counts_processed = train_df['Processed_Text'].str.split().str.len() # Corrected column name
char_counts_processed = train_df['Processed_Text'].str.len()             # Corrected column name
avg_word_processed = word_counts_processed.mean()
avg_char_processed = char_counts_processed.mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Unprocessed Plot
axes[0].bar(['Avg Words', 'Avg Chars'], [avg_word_unprocessed , avg_char_unprocessed], color=['tomato', 'steelblue'])
axes[0].set_title('Unprocessed Tweets')
axes[0].set_ylabel('Average Count')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate([avg_word_unprocessed, avg_char_unprocessed]):
    axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')


# Processed Plot
axes[1].bar(['Avg Words', 'Avg Chars'], [avg_word_processed, avg_char_processed], color=['tomato', 'steelblue'])
axes[1].set_title('Processed Tweets')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate([avg_word_processed, avg_char_processed]):
    axes[1].text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')

fig.suptitle('Average Word and Character Count per Tweet', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Used for non-final models
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

BERT_MAX_LEN = 128

# 3) Train Set Distribution of Sequence Lengths (after BERT Tokenization)
bert_token_lengths_train = []
for txt in train_df['Processed_Text']: # Using processed text
    input_ids = tokenizer.encode(txt, add_special_tokens=True)
    bert_token_lengths_train.append(len(input_ids))

plt.figure(figsize=(10, 6))
plt.hist(bert_token_lengths_train, bins=50, alpha=0.7, color='skyblue', label='Train Set Token Lengths')
# If you calculated for all: plt.hist(bert_token_lengths_all, bins=50, alpha=0.7, color='skyblue', label='All Sentences Token Lengths')
plt.axvline(BERT_MAX_LEN, color='red', linestyle='dashed', linewidth=2, label=f'BERT_MAX_LEN = {BERT_MAX_LEN}')
plt.title('Train Set Distribution of BERT Tokenized Sequence Lengths')
plt.xlabel('Number of Tokens (after BERT tokenization)')
plt.ylabel('Number of Sentences')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3) Validation Set Distribution of Sequence Lengths (after BERT Tokenization)
bert_token_lengths_train = []
for txt in val_df['Processed_Text']: # Using processed text
    input_ids = tokenizer.encode(txt, add_special_tokens=True)
    bert_token_lengths_train.append(len(input_ids))

plt.figure(figsize=(10, 6))
plt.hist(bert_token_lengths_train, bins=50, alpha=0.7, color='skyblue', label='Train Set Token Lengths')
# If you calculated for all: plt.hist(bert_token_lengths_all, bins=50, alpha=0.7, color='skyblue', label='All Sentences Token Lengths')
plt.axvline(BERT_MAX_LEN, color='red', linestyle='dashed', linewidth=2, label=f'BERT_MAX_LEN = {BERT_MAX_LEN}')
plt.title('Validation Set Distribution of BERT Tokenized Sequence Lengths')
plt.xlabel('Number of Tokens (after BERT tokenization)')
plt.ylabel('Number of Sentences')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

def bert_encode_data(sentences, labels, model_max_len):
    # Tokenize all of the sentences and map the tokens to their word IDs.
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
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    if labels is not None:
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    return input_ids, attention_masks

x_train_ids, x_train_masks, y_train_tensor = bert_encode_data(sentences_train, labels_train, BERT_MAX_LEN)
x_val_ids, x_val_masks, y_val_tensor = bert_encode_data(sentences_val, labels_val, BERT_MAX_LEN)
x_test_ids, x_test_masks = bert_encode_data(sentences_test, None, BERT_MAX_LEN)

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

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
    hidden_dropout_prob = 0.15,
    attention_probs_dropout_prob = 0.1
)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr = 2.9609614936542493e-05, weight_decay= 0.1, eps = 1e-8)

epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

warmup_ratio = 0.05
num_warmup_steps_val = int(total_steps * warmup_ratio)

# Create the learning rate scheduler to dynamically adjust the learning rate.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps_val, num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function that takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Training Loop
training_stats = []
best_val_accuracy = 0.0
no_improve_epochs_bert = 0
early_stop_epochs_bert = 2

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
                       token_type_ids=None,
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
                           token_type_ids=None,
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

    # Early stopping for BERT
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), 'best_bert_model.pth')
        print("  New best BERT model saved!")
        no_improve_epochs_bert = 0
    else:
        no_improve_epochs_bert += 1
        if no_improve_epochs_bert >= early_stop_epochs_bert:
            print(f"  Early stopping BERT after {early_stop_epochs_bert} epochs without improvement.")
            break

    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current LR: {current_lr:.2e}")

print("\nTraining complete!")
print(f"Total training took {format_time(time.time()-total_t0)}")

# Load the best model weights
model.load_state_dict(torch.load('best_bert_model.pth'))

model.eval()
all_val_preds_bert = []
all_val_true_bert = []
all_val_probs_bert = []

with torch.no_grad():
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits
        
        # Get probabilities for ROC curve
        probs = torch.softmax(logits, dim=1)[:, 1] # Probability of positive class
        all_val_probs_bert.extend(probs.cpu().numpy())

        preds = torch.argmax(logits, dim=1)
        all_val_preds_bert.extend(preds.cpu().numpy())
        all_val_true_bert.extend(b_labels.cpu().numpy())

val_preds_np_bert = np.array(all_val_preds_bert)
val_true_np_bert = np.array(all_val_true_bert)
val_probs_np_bert = np.array(all_val_probs_bert)

accuracy_bert = accuracy_score(val_true_np_bert, val_preds_np_bert)
precision_bert = precision_score(val_true_np_bert, val_preds_np_bert)
recall_bert = recall_score(val_true_np_bert, val_preds_np_bert)
f1_bert = f1_score(val_true_np_bert, val_preds_np_bert)

print("\nFinal BERT Model Evaluation Metrics (on Validation Set):")
print(f"Accuracy: {accuracy_bert:.4f}")
print(f"Precision: {precision_bert:.4f}")
print(f"Recall: {recall_bert:.4f}")
print(f"F1 Score: {f1_bert:.4f}")
print("\nClassification Report (BERT):")
print(classification_report(val_true_np_bert, val_preds_np_bert))

model.eval()
test_predictions_bert = []
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        test_predictions_bert.extend(preds.cpu().numpy())

test_preds_np_bert = np.array(test_predictions_bert).flatten().astype(int)

# Plots
df_stats_bert = pd.DataFrame(data=training_stats)
df_stats_bert = df_stats_bert.set_index('epoch')

# Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(df_stats_bert['Valid. Accur.'], 'g-o', label="Validation Accuracy (BERT)")
plt.plot(df_stats_bert['Training Loss'], 'b-o', label="Training Loss (BERT)")
plt.plot(df_stats_bert['Valid. Loss'], 'r-o', label="Validation Loss (BERT)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Loss")
plt.title("BERT Training & Validation Accuracy and Loss")
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ROC Curve
fpr_bert, tpr_bert, _ = roc_curve(val_true_np_bert, val_probs_np_bert)
roc_auc_bert = auc(fpr_bert, tpr_bert)
plt.figure(figsize=(6, 5))
plt.plot(fpr_bert, tpr_bert, color='darkorange', lw=2, label=f'BERT ROC curve (AUC = {roc_auc_bert:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - BERT')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm_bert = confusion_matrix(val_true_np_bert, val_preds_np_bert)
disp_bert = ConfusionMatrixDisplay(confusion_matrix=cm_bert, display_labels=["Negative", "Positive"])
disp_bert.plot(cmap=plt.cm.Greens) # Changed color for distinction
plt.title('Confusion Matrix - BERT')
plt.grid(False)
plt.tight_layout()
plt.show()

# Create submission file
submission_df_bert = pd.DataFrame({'ID': test_df['ID'], 'Label': test_preds_np_bert})
submission_df_bert.to_csv("submission_bert.csv", index=False)
print("BERT Test set predictions saved to 'submission_bert.csv'")