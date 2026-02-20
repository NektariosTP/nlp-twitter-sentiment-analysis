# Import libraries
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

import gensim
import gensim.downloader as api

import re
import string

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import  roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

import optuna

# Set random seeds for reproducibility
torch.manual_seed(10)
np.random.seed(10)

# Load pre-trained embedding model
word2vec_model = api.load("glove-twitter-200")

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

    # Remove hashtags
    text = re.sub(r"#\S+", "X", text)

    # Remove punctuation
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans(punctuation_to_remove, " " * len(punctuation_to_remove)))
    text = re.sub(r"\b(\w+)'(\w+)\b", r"\1\2", text)

    # Remove non-ASCII characters (fix mojibake issues)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove repeated characters (3 or more times → 1 occurrence)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

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

train_df['Processed'] = train_df['Text'].astype(str).apply(preprocess_text)

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
word_counts_unprocessed = train_df['Text'].str.split().str.len()
char_counts_unprocessed = train_df['Text'].str.len()
avg_word_unprocessed = word_counts_unprocessed.mean()
avg_char_unprocessed = char_counts_unprocessed.mean()
word_counts_processed = train_df['Processed'].str.split().str.len()
char_counts_processed = train_df['Processed'].str.len()
avg_word_processed = word_counts_processed.mean()
avg_char_processed = char_counts_processed.mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Unprocessed Plot
axes[0].bar(['Avg Words', 'Avg Chars'], [avg_word_unprocessed , avg_char_unprocessed], color=['tomato', 'steelblue'])
axes[0].set_title('Unprocessed Tweets')
axes[0].set_ylabel('Average Count')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
# Processed Plot
axes[1].bar(['Avg Words', 'Avg Chars'], [avg_word_processed, avg_char_processed], color=['tomato', 'steelblue'])
axes[1].set_title('Processed Tweets')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

fig.suptitle('Average Word and Character Count per Tweet', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 3) Vocabulary Size Comparison - Unprocessed and Processed
raw_tokens = train_df['Text'].str.split().explode().dropna()
proc_tokens = train_df['Processed'].str.split().explode().dropna()
vocab_before = raw_tokens.nunique()
vocab_after  = proc_tokens.nunique()
plt.figure()
plt.bar(['Before', 'After'], [vocab_before, vocab_after])
plt.ylabel('Unique Tokens')
plt.title('Vocabulary Size: Before vs After Preprocessing')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 4) Out-of-Vocabulary (OOV) Rate - Unprocessed and Processed
emb_vocab = set(word2vec_model.key_to_index.keys())

# Unprocessed
raw_vocab = set(raw_tokens.unique())
oov_rate_unprocessed = len(raw_vocab - emb_vocab) / len(raw_vocab)
# Processed
proc_vocab = set(proc_tokens.unique())
oov_rate_processed = len(proc_vocab - emb_vocab) / len(proc_vocab)

plt.figure(figsize=(6, 5))
plt.bar(['Unprocessed', 'Processed'], [oov_rate_unprocessed, oov_rate_processed], color=['tomato', 'steelblue'])
plt.ylim(0, 1)
plt.ylabel('OOV Rate')
plt.title('Out-of-Vocabulary Rate Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 5) Token Frequency Distribution (Top N) - Processed
N = 20
top_tokens = proc_tokens.value_counts().head(N)
plt.figure(figsize=(10, 6))
top_tokens.plot(kind='bar')
plt.xlabel('Token')
plt.ylabel('Frequency')
plt.title(f'Top {N} Tokens Frequency Distribution')
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply preprocessing
train_df["Text"] = train_df['Processed']
val_df["Text"] = val_df["Text"].apply(preprocess_text)
test_df["Text"] = test_df["Text"].apply(preprocess_text)

# Separate the data
x_train, y_train = train_df["Text"], train_df["Label"]
x_val, y_val = val_df["Text"], val_df["Label"]
x_test = test_df["Text"]

# Word2Vec model
vector_size = word2vec_model.vector_size

def sentence_to_vec(sentence, model, vector_size):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    
    if len(word_vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(word_vectors, axis=0)

# Convert dataset to numerical form
x_train_w2v = np.array([sentence_to_vec(text, word2vec_model, vector_size) for text in x_train])
x_val_w2v = np.array([sentence_to_vec(text, word2vec_model, vector_size) for text in x_val])
x_test_w2v = np.array([sentence_to_vec(text, word2vec_model, vector_size) for text in x_test])

# Save in tensors
x_train_tensor = torch.tensor(x_train_w2v, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val_w2v, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_w2v, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# Create Neural Network
class Net(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out, dropout_prob, activation_fn):
        super(Net, self).__init__()

        self.activation = activation_fn()
        
        self.linear1 = nn.Linear(D_in, H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.linear2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.linear3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(H3)
        
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        h1 = self.linear1(x)
        h1 = self.bn1(h1)
        h1 = self.activation(h1)
        h1 = self.dropout1(h1)
        
        h2 = self.linear2(h1)
        h2 = self.bn2(h2)
        h2 = self.activation(h2)
        h2 = self.dropout2(h2)
        
        h3 = self.linear3(h2)
        h3 = self.bn3(h3)
        h3 = self.activation(h3)
        
        out = self.linear4(h3)
        return out
    
# Define layer sizes
D_in = vector_size
H1, H2, H3 = 500, 201, 65
D_out = 1
dropout_prob = 0.2895417180593599
learning_rate = 2e-4
activation_fn = nn.GELU

# Initialise model, loss, optimizer
model = Net(D_in, H1, H2, H3, D_out, dropout_prob, activation_fn)
loss_func = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize ReduceLROnPlateau scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',       # Monitor validation accuracy
    # if the validation accuracy does not improve by at least 0.5% over three consecutive epochs, the scheduler considers this a plateau and the LR is multiplied by factor=0.1
    factor=0.1,       # Reduce LR by 10x
    patience=3,       # Wait 3 epochs w/o improvement
    threshold=0.005,
)

# Initialise dataloader
dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# Training loop
epochs = 30
best_val_accuracy = 0.0
no_improve = 0
early_stop=5

# Used in plots
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_func(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        train_outputs = model(x_train_tensor)
        train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
        train_acc = (train_preds == y_train_tensor).float().mean()
        
        val_outputs = model(x_val_tensor)
        val_loss = loss_func(val_outputs, y_val_tensor) / len(val_df)
        val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
        val_acc = (val_preds == y_val_tensor).float().mean()
    
    avg_loss = total_loss / len(train_df)
    current_lr = optimizer.param_groups[0]['lr']

    # Used in plots
    train_losses.append(avg_loss)
    val_losses.append(val_loss.item())
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc.item())
    
    print(f"Epoch {epoch+1}:")
    print(f"  Train Accuracy: {train_acc.item()*100:.2f}%")
    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Val Accuracy: {val_acc.item()*100:.2f}%")
    print(f"  Val Loss: {val_loss.item():.4f}")
    print(f"  Current LR: {current_lr:.2e}")

    # Early stopping check
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("  New best model!")
        no_improve = 0
    else:
        no_improve += 1
        # If the validation performance doesn’t improve over several epochs --> model has likely converged or is stuck and stops further training
        if no_improve > early_stop:
            print(f"  Early stopping after {early_stop} epochs without improvement")
            break

    if epoch >= 3:
        scheduler.step(val_acc)
        if optimizer.param_groups[0]['lr'] != current_lr:
            print(f"  LR reduced to {optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"  No LR change (best acc: {best_val_accuracy:.4f}%)")

# Load the best model weights
model.load_state_dict(torch.load('best_model.pth'))

# Final Evaluation
model.eval()
with torch.no_grad():
    # Get predictions for validation set
    val_outputs = model(x_val_tensor)
    val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
    
    # Convert to numpy for sklearn metrics
    val_preds_np = val_preds.cpu().numpy().flatten()
    val_true_np = y_val_tensor.cpu().numpy().flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(val_true_np, val_preds_np)
    precision = precision_score(val_true_np, val_preds_np)
    recall = recall_score(val_true_np, val_preds_np)
    f1 = f1_score(val_true_np, val_preds_np)
    
    # Print metrics
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_true_np, val_preds_np))

    # Generate predictions for test set
    test_outputs = model(x_test_tensor)
    test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
    test_preds_np = test_preds.cpu().numpy().flatten().astype(int)

# Learning Curve
plt.figure(figsize=(10, 6))

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Loss")
plt.title("Training & Validation Accuracy and Loss")
plt.ylim(0, 1)  # Set y-axis from 0 to 1
plt.yticks(np.arange(0, 1.1, 0.2))  # Tick marks every 0.2
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ROC Curve
val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
fpr, tpr, thresholds = roc_curve(val_true_np, val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(val_true_np, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.grid(False)
plt.tight_layout()
plt.show()

# Create submission file
submission_df = pd.DataFrame({'ID': test_df['ID'], 'Label': test_preds_np})
submission_df.to_csv("submission.csv", index=False)

print("Test set predictions saved to 'submission.csv'")





# Below are the results of all Optuna Tests executed
#
#     H1 = trial.suggest_int('H1', 64, 512)
#     H2 = trial.suggest_int('H2', 32, 256)
#     H3 = trial.suggest_int('H3', 16, 128)
#     H4 = trial.suggest_int('H4', 8, 64)
#     dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
#     learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

# [I 2025-04-14 17:35:07,662] Trial 0 finished with value: 0.7603311538696289 and parameters: {'H1': 271, 'H2': 37, 'H3': 79, 'H4': 53, 'dropout_prob': 0.29345592199549764, 'lr': 0.00013268810437969312}. Best is trial 0 with value: 0.7603311538696289.
# [I 2025-04-14 17:38:09,855] Trial 1 finished with value: 0.7668883800506592 and parameters: {'H1': 407, 'H2': 65, 'H3': 63, 'H4': 51, 'dropout_prob': 0.10417591067289839, 'lr': 0.0002646949522764922}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:40:34,828] Trial 2 finished with value: 0.7611567378044128 and parameters: {'H1': 92, 'H2': 48, 'H3': 29, 'H4': 55, 'dropout_prob': 0.14959339745587932, 'lr': 0.00016712172621323956}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:43:15,324] Trial 3 finished with value: 0.7466506361961365 and parameters: {'H1': 93, 'H2': 77, 'H3': 27, 'H4': 63, 'dropout_prob': 0.48249170291407784, 'lr': 3.404739640217067e-05}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:45:17,661] Trial 4 finished with value: 0.7578781247138977 and parameters: {'H1': 86, 'H2': 100, 'H3': 125, 'H4': 27, 'dropout_prob': 0.10688153617579116, 'lr': 0.00010825998689065124}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:47:42,346] Trial 5 finished with value: 0.7657326459884644 and parameters: {'H1': 176, 'H2': 40, 'H3': 126, 'H4': 52, 'dropout_prob': 0.1672646405700935, 'lr': 0.0007235965274063964}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:49:58,283] Trial 6 finished with value: 0.7593405246734619 and parameters: {'H1': 381, 'H2': 117, 'H3': 69, 'H4': 40, 'dropout_prob': 0.21467551830480877, 'lr': 6.372782791383326e-05}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:52:34,358] Trial 7 finished with value: 0.7555901408195496 and parameters: {'H1': 259, 'H2': 129, 'H3': 83, 'H4': 10, 'dropout_prob': 0.49735416395305043, 'lr': 7.393132739325845e-05}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:55:33,424] Trial 8 finished with value: 0.7457779049873352 and parameters: {'H1': 97, 'H2': 241, 'H3': 77, 'H4': 54, 'dropout_prob': 0.4523539309158964, 'lr': 2.4154175402579904e-05}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 17:58:56,191] Trial 9 finished with value: 0.7539626359939575 and parameters: {'H1': 367, 'H2': 223, 'H3': 40, 'H4': 19, 'dropout_prob': 0.3695866488766181, 'lr': 3.0623272581776474e-05}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 18:02:28,258] Trial 10 finished with value: 0.7664874196052551 and parameters: {'H1': 479, 'H2': 187, 'H3': 54, 'H4': 41, 'dropout_prob': 0.24803962183432113, 'lr': 0.00047642778802176684}. Best is trial 1 with value: 0.7668883800506592.
# [I 2025-04-14 18:05:28,805] Trial 11 finished with value: 0.7672893404960632 and parameters: {'H1': 508, 'H2': 181, 'H3': 56, 'H4': 40, 'dropout_prob': 0.24839958981290883, 'lr': 0.00047878203878297957}. Best is trial 11 with value: 0.7672893404960632.
# [I 2025-04-14 18:09:03,787] Trial 12 finished with value: 0.7651901245117188 and parameters: {'H1': 512, 'H2': 178, 'H3': 100, 'H4': 34, 'dropout_prob': 0.31098130915607214, 'lr': 0.00032230498371121934}. Best is trial 11 with value: 0.7672893404960632.
# [I 2025-04-14 18:12:24,897] Trial 13 finished with value: 0.7677611112594604 and parameters: {'H1': 429, 'H2': 167, 'H3': 56, 'H4': 45, 'dropout_prob': 0.10139369892414993, 'lr': 0.0002996820474465651}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:15:44,277] Trial 14 finished with value: 0.7406123280525208 and parameters: {'H1': 451, 'H2': 168, 'H3': 48, 'H4': 30, 'dropout_prob': 0.3710115634245882, 'lr': 1.0399724436207625e-05}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:18:56,417] Trial 15 finished with value: 0.7664638161659241 and parameters: {'H1': 341, 'H2': 207, 'H3': 16, 'H4': 45, 'dropout_prob': 0.19105150706689994, 'lr': 0.000832753097535303}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:22:06,458] Trial 16 finished with value: 0.7668176293373108 and parameters: {'H1': 451, 'H2': 151, 'H3': 96, 'H4': 23, 'dropout_prob': 0.2508587699318912, 'lr': 0.00027423018822409954}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:25:18,106] Trial 17 finished with value: 0.7642937898635864 and parameters: {'H1': 321, 'H2': 151, 'H3': 58, 'H4': 45, 'dropout_prob': 0.361496830051354, 'lr': 0.0005465780431305136}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:28:42,262] Trial 18 finished with value: 0.7661572098731995 and parameters: {'H1': 422, 'H2': 196, 'H3': 36, 'H4': 64, 'dropout_prob': 0.1390044466090122, 'lr': 0.00021006241898287103}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:31:52,934] Trial 19 finished with value: 0.7658741474151611 and parameters: {'H1': 506, 'H2': 256, 'H3': 49, 'H4': 35, 'dropout_prob': 0.23035696897240732, 'lr': 0.00041407729408490145}. Best is trial 13 with value: 0.7677611112594604.
# [I 2025-04-14 18:35:12,628] Trial 20 finished with value: 0.7663930654525757 and parameters: {'H1': 445, 'H2': 219, 'H3': 92, 'H4': 47, 'dropout_prob': 0.3189156012742088, 'lr': 0.0008385655373394957}. Best is trial 13 with value: 0.7677611112594604.

#     H1 = trial.suggest_int('H1', 400, 512)
#     H2 = trial.suggest_int('H2', 150, 256)
#     H3 = trial.suggest_int('H3', 32, 96)
#     H4 = trial.suggest_int('H4', 24, 64)
#     dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.3)
#     learning_rate = trial.suggest_float('lr', 2e-4, 8e-4, log=True)

# [I 2025-04-15 19:57:08,840] Trial 0 finished with value: 0.7665581703186035 and parameters: {'H1': 403, 'H2': 154, 'H3': 35, 'H4': 55, 'dropout_prob': 0.10404511969744244, 'lr': 0.0004057226285864009}. Best is trial 0 with value: 0.7665581703186035.
# [I 2025-04-15 19:59:45,196] Trial 1 finished with value: 0.7638456225395203 and parameters: {'H1': 485, 'H2': 165, 'H3': 68, 'H4': 57, 'dropout_prob': 0.28383345517700753, 'lr': 0.0005730395314205658}. Best is trial 0 with value: 0.7665581703186035.
# [I 2025-04-15 20:03:35,137] Trial 2 finished with value: 0.76764315366745 and parameters: {'H1': 408, 'H2': 174, 'H3': 72, 'H4': 59, 'dropout_prob': 0.10636730480531482, 'lr': 0.0003246689363509672}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:07:46,360] Trial 3 finished with value: 0.7666053175926208 and parameters: {'H1': 508, 'H2': 188, 'H3': 66, 'H4': 56, 'dropout_prob': 0.22213328800404436, 'lr': 0.00024874266987586077}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:11:59,498] Trial 4 finished with value: 0.7656618356704712 and parameters: {'H1': 497, 'H2': 178, 'H3': 90, 'H4': 59, 'dropout_prob': 0.15016914840086887, 'lr': 0.0003082332445595711}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:16:03,675] Trial 5 finished with value: 0.7658033967018127 and parameters: {'H1': 512, 'H2': 219, 'H3': 83, 'H4': 49, 'dropout_prob': 0.16876763587451868, 'lr': 0.00036941797678229885}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:19:53,608] Trial 6 finished with value: 0.76320880651474 and parameters: {'H1': 418, 'H2': 169, 'H3': 34, 'H4': 28, 'dropout_prob': 0.23457942762764966, 'lr': 0.0002786927465253458}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:24:08,611] Trial 7 finished with value: 0.7644588947296143 and parameters: {'H1': 412, 'H2': 246, 'H3': 77, 'H4': 40, 'dropout_prob': 0.23713386713168988, 'lr': 0.0003008318785306086}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:27:40,652] Trial 8 finished with value: 0.766039252281189 and parameters: {'H1': 414, 'H2': 212, 'H3': 72, 'H4': 60, 'dropout_prob': 0.2004218306571997, 'lr': 0.0002601825625091611}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:30:28,073] Trial 9 finished with value: 0.7638456225395203 and parameters: {'H1': 486, 'H2': 192, 'H3': 45, 'H4': 60, 'dropout_prob': 0.26038371515714565, 'lr': 0.000410932186042337}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:34:22,890] Trial 10 finished with value: 0.7664874196052551 and parameters: {'H1': 444, 'H2': 230, 'H3': 52, 'H4': 42, 'dropout_prob': 0.10266007831251803, 'lr': 0.00020549941413800703}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:38:10,963] Trial 11 finished with value: 0.7646239995956421 and parameters: {'H1': 452, 'H2': 196, 'H3': 56, 'H4': 50, 'dropout_prob': 0.1892512258547229, 'lr': 0.0002049773117843865}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:41:07,499] Trial 12 finished with value: 0.7653552293777466 and parameters: {'H1': 432, 'H2': 185, 'H3': 64, 'H4': 63, 'dropout_prob': 0.15325966776386, 'lr': 0.0005517058649982576}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:44:20,080] Trial 13 finished with value: 0.7650721669197083 and parameters: {'H1': 466, 'H2': 151, 'H3': 94, 'H4': 50, 'dropout_prob': 0.213488893912194, 'lr': 0.0007033017454371969}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:48:28,276] Trial 14 finished with value: 0.7653552293777466 and parameters: {'H1': 465, 'H2': 206, 'H3': 58, 'H4': 34, 'dropout_prob': 0.1369924166145284, 'lr': 0.0002532724463296767}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:51:44,501] Trial 15 finished with value: 0.7631380558013916 and parameters: {'H1': 433, 'H2': 175, 'H3': 78, 'H4': 53, 'dropout_prob': 0.2966830239454959, 'lr': 0.0003454714726682281}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 20:55:51,536] Trial 16 finished with value: 0.7639635801315308 and parameters: {'H1': 475, 'H2': 188, 'H3': 63, 'H4': 46, 'dropout_prob': 0.2287150829878161, 'lr': 0.00022822604993413795}. Best is trial 2 with value: 0.76764315366745.
# [I 2025-04-15 21:00:00,395] Trial 17 finished with value: 0.7675252556800842 and parameters: {'H1': 511, 'H2': 161, 'H3': 84, 'H4': 62, 'dropout_prob': 0.1814491869123047, 'lr': 0.00048197828267824023}. Best is trial 2 with value: 0.76764315366745.

#     H1 = trial.suggest_int('H1', 400, 512)
#     H2 = trial.suggest_int('H2', 128, min(H1, 256))
#     H3 = trial.suggest_int('H3', 32, min(H2, 96))
#     dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.3)
#     learning_rate = trial.suggest_float('lr', 2e-4, 8e-4, log=True)

# [I 2025-04-15 21:17:41,624] Trial 0 finished with value: 0.7647891044616699 and parameters: {'H1': 427, 'H2': 168, 'H3': 77, 'dropout_prob': 0.19925318152950844, 'lr': 0.0002046787143253785}. Best is trial 0 with value: 0.7647891044616699.
# [I 2025-04-15 21:20:38,112] Trial 1 finished with value: 0.7657090425491333 and parameters: {'H1': 424, 'H2': 178, 'H3': 49, 'dropout_prob': 0.18450746600502438, 'lr': 0.0006468884973308204}. Best is trial 1 with value: 0.7657090425491333.
# [I 2025-04-15 21:24:15,609] Trial 2 finished with value: 0.7663694620132446 and parameters: {'H1': 445, 'H2': 224, 'H3': 53, 'dropout_prob': 0.16840457757837046, 'lr': 0.000786870953306065}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:27:33,611] Trial 3 finished with value: 0.7663458585739136 and parameters: {'H1': 502, 'H2': 155, 'H3': 84, 'dropout_prob': 0.1850430572722076, 'lr': 0.00045645407592288437}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:30:31,060] Trial 4 finished with value: 0.7648834586143494 and parameters: {'H1': 488, 'H2': 166, 'H3': 34, 'dropout_prob': 0.26031479418818326, 'lr': 0.00023876256806603565}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:33:50,919] Trial 5 finished with value: 0.7656382918357849 and parameters: {'H1': 464, 'H2': 224, 'H3': 69, 'dropout_prob': 0.22285553529369959, 'lr': 0.0002590566171847744}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:36:30,924] Trial 6 finished with value: 0.7648599147796631 and parameters: {'H1': 476, 'H2': 236, 'H3': 38, 'dropout_prob': 0.23303820810059386, 'lr': 0.0003552302175192481}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:39:11,456] Trial 7 finished with value: 0.765425980091095 and parameters: {'H1': 476, 'H2': 212, 'H3': 78, 'dropout_prob': 0.13683345928256413, 'lr': 0.00031199549914612284}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:41:33,779] Trial 8 finished with value: 0.7641758918762207 and parameters: {'H1': 412, 'H2': 134, 'H3': 55, 'dropout_prob': 0.16198404021946175, 'lr': 0.00026256242719198045}. Best is trial 2 with value: 0.7663694620132446.
# [I 2025-04-15 21:45:02,456] Trial 9 finished with value: 0.7670770883560181 and parameters: {'H1': 447, 'H2': 196, 'H3': 49, 'dropout_prob': 0.12492778938203802, 'lr': 0.00036965479436828965}. Best is trial 9 with value: 0.7670770883560181.
# [I 2025-04-15 21:48:41,810] Trial 10 finished with value: 0.7679498195648193 and parameters: {'H1': 443, 'H2': 254, 'H3': 66, 'dropout_prob': 0.10215812325616692, 'lr': 0.0004960806771545662}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 21:52:26,237] Trial 11 finished with value: 0.7671478390693665 and parameters: {'H1': 448, 'H2': 256, 'H3': 94, 'dropout_prob': 0.10193613042497189, 'lr': 0.0004888726830507955}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 21:56:06,795] Trial 12 finished with value: 0.7669827342033386 and parameters: {'H1': 438, 'H2': 256, 'H3': 92, 'dropout_prob': 0.10013605914486715, 'lr': 0.0005059410028161044}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 21:59:01,337] Trial 13 finished with value: 0.767053484916687 and parameters: {'H1': 403, 'H2': 252, 'H3': 94, 'dropout_prob': 0.29807571794004234, 'lr': 0.0005618141672208533}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 22:02:38,183] Trial 14 finished with value: 0.7674545049667358 and parameters: {'H1': 454, 'H2': 241, 'H3': 66, 'dropout_prob': 0.10614261671974781, 'lr': 0.0004399304539195594}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 22:05:31,873] Trial 15 finished with value: 0.7674545049667358 and parameters: {'H1': 464, 'H2': 235, 'H3': 64, 'dropout_prob': 0.13860020236786605, 'lr': 0.0004076105693036385}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 22:07:58,164] Trial 16 finished with value: 0.7657090425491333 and parameters: {'H1': 435, 'H2': 201, 'H3': 64, 'dropout_prob': 0.12889621979451202, 'lr': 0.0005864375603304399}. Best is trial 10 with value: 0.7679498195648193.
# [I 2025-04-15 22:10:30,527] Trial 17 finished with value: 0.7660156488418579 and parameters: {'H1': 465, 'H2': 238, 'H3': 71, 'dropout_prob': 0.15534568784397737, 'lr': 0.0007355190996613152}. Best is trial 10 with value: 0.7679498195648193.
