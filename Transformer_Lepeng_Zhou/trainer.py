
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# load data from final_df.csv
final_df = pd.read_csv('final_df_balanced.csv') 

# load the test data
test_df = pd.read_csv('final_df_test.csv')

# import torch
import torch
import torch.nn as nn
print(torch.cuda.is_available())
class InputEmbedding(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embed_size) # i still don't know why we multiply by sqrt of embed_size 
        

class PositionEncoding(nn.Module):

    def __init__(self, embed_size: int, max_len: int, dropout: float):
        super().__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # create a zero matrix 
        pe = torch.zeros(max_len, embed_size)
        # create the pos vector
        pos = torch.arange(0, max_len).unsqueeze(1)
        # calculate the div term
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(np.log(10000.0) / embed_size))
        # calculate the pos for even numbers starts from 0
        pe[:, 0::2] = torch.sin(pos * div_term)
        # calculate the pos for odd numbers starts from 1
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0) 

        # save the pe as along with model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # fixed

        return self.dropout(x)  # prevent overfitting


# this class is crucial since we will use it many times
class LayerNormalizing(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # change 1 to embed_size if needed
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        norm_x = (x - mean) / (std + self.eps)
        return self.alpha * norm_x  + self.bias


class ResConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizing()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResConnection(dropout), ResConnection(dropout)])
    
    def forward(self, x, src_mask):
        # First residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, attn_mask=src_mask)[0])
        # Second residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# here we serialize the encoder blocks
class EncoderChains(nn.Module):

    def __init__(self, encoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = LayerNormalizing()
    
    def forward(self, x, src_mask):
        for block in self.encoder_blocks:
            x = block(x, src_mask)

        # finally normalize the output once again
        return self.norm(x)


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, max_len, vocab_size, num_classes, dropout_rate, block_num):
        super().__init__()
        
        # Input Embedding layer
        self.embedding = InputEmbedding(embed_size, vocab_size)
        
        # Positional Encoding layer
        self.positional_encoding = PositionEncoding(embed_size, max_len, dropout_rate)
        
        # Initialize the encoder blocks
        self.encoder_modulelist = nn.ModuleList()
        for _ in range(block_num):
            self_attention_block = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout_rate)
            feed_forward_block = nn.Sequential(
                nn.Linear(embed_size, ff_hidden_size),
                nn.ReLU(),
                nn.Linear(ff_hidden_size, embed_size)
            )
            encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout_rate)
            self.encoder_modulelist.append(encoder_block)
        
        self.encoder = EncoderChains(self.encoder_modulelist) 

        # conv layers
        self.cnn_layer1 = nn.Conv1d(in_channels=embed_size, out_channels=int(embed_size), kernel_size=2, padding=0)
        self.cnn_layer2 = nn.Conv1d(in_channels=embed_size, out_channels=int(embed_size), kernel_size=3, padding=1)
        self.cnn_layer3 = nn.Conv1d(in_channels=embed_size, out_channels=int(embed_size), kernel_size=4, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dense_1 = nn.Linear(3 * int(embed_size), int(embed_size*2))
        self.dropout = nn.Dropout(dropout_rate)
        self.last_dense = nn.Linear(int(embed_size*2), num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, src_mask=None):
        # Input embedding
        x = self.embedding(x)
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Encoder
        x = self.encoder(x, src_mask)
        
        
        # convolutional layer
        x = x.permute(0, 2, 1)
        l_1 = self.pool(nn.functional.relu(self.cnn_layer1(x))).squeeze(2)
        l_2 = self.pool(nn.functional.relu(self.cnn_layer2(x))).squeeze(2)
        l_3 = self.pool(nn.functional.relu(self.cnn_layer3(x))).squeeze(2)
       

        # Concatenate the outputs of the convolutional layers
        concatenated = torch.cat([l_1, l_2, l_3], axis=-1)
        concatenated = nn.functional.relu(self.dense_1(concatenated))
        concatenated = self.dropout(concatenated)

        # Fully connected layers
        y_hat = self.softmax(self.last_dense(concatenated))

        return y_hat

# Hyper Parameters
#****************************************************
embed_size = 64
num_heads = 16
ff_hidden_size = 2 * embed_size
max_len = 340
vocab_size = 55810
num_classes = 2
dropout_rate = 0.1
block_num = 3
train_batch_sz = 64
#****************************************************



num_epochs = 100



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = TransformerEncoderClassifier(embed_size, num_heads, ff_hidden_size, max_len, vocab_size, num_classes, dropout_rate, block_num).to(device)

# find the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Prepare data for training and testing
final_df['Text'] = final_df['Text'].apply(lambda x: x.split())

test_df['Text'] = test_df['Text'].apply(lambda x: x.split())

# Convert y to one hot encoding
# Result that 1 should be scam, 0 should be normal
y_train = pd.get_dummies(final_df['Result'])
y_test = pd.get_dummies(test_df['Result'])

# TODOs
# handle misspelled words
# class imbalance problem? figure out a way to give scam class more weight


# save and load the tokenizer
import pickle

# Initialize the tokenizer with the OOV token
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')

# Fit the tokenizer on the texts
tokenizer.fit_on_texts(final_df['Text'])

# Convert texts to sequences of int
X_train = tokenizer.texts_to_sequences(final_df['Text'])

# Pad the sequences to ensure all are the same length
X_train = pad_sequences(X_train, maxlen=max_len)

# Convert texts to sequences of int
X_test = tokenizer.texts_to_sequences(test_df['Text'])

# Pad the sequences to ensure all are the same length
X_test = pad_sequences(X_test, maxlen=max_len)




# save 
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


# # Load the tokenizer from the file
# with open('tokenizer.pkl', 'rb') as f:
#     loaded_tokenizer = pickle.load(f)


# see the dimension of the data
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Convert to tensor and move to device
X_train = torch.tensor(X_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.long).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float).to(device)

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# validation split
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=train_batch_sz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# init a list to store the losses
train_losses = []
val_losses = []

# Early stopping parameters
early_stopping_patience = 3
early_stopping_min_delta = 0.0001

best_val_loss = float('inf')
patience_counter = 0

# set 1 to train, 0 to load the model and evaluate
mode = 1

if mode == 1:

    import time
    start = time.time()

    # Training loop
    best_accuracy = 0
    no_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0
        last_min = start
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 100 == 99:  # Print every 100 batches
                mins = (time.time() - start)
                # calculate the time for each load
                #print(f'Time/load: {(mins - last_min) / 60:.4f}')
                last_min = mins
                progress_percent = (i + 1) / len(train_loader) * 100
                print(f'\rEpoch [{epoch + 1}/{num_epochs}], Progress : {progress_percent:.1f}, Loss: {running_loss / 100:.4f}, Trained minutes: {(mins/60):.1f}', end='\r')
                running_loss = 0.0

        train_loss /= len(train_loader)

        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check early stopping conditions
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'transformer_encoder_classifier.pth')
            print("Model saved, re-setting patience counter to zero")
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}")

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            print(f"Best Validation Loss: {best_val_loss:.4f}")
            break
        

else:

    # load the model
    model = TransformerEncoderClassifier(embed_size, num_heads, ff_hidden_size, max_len, vocab_size, num_classes, dropout_rate, block_num).to(device)
    model.load_state_dict(torch.load('transformer_encoder_classifier.pth'))

    model.eval()

    # Initialize lists to store true labels and predictions
    y_pred = []
    y_true = []
    y_prob = []

    # Make predictions and collect true labels and predicted probabilities
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, axis=1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(outputs[:, 1].cpu().numpy())  # Collect the probabilities of the positive class

    # Convert to numpy arrays for metric calculations
    y_pred = np.array(y_pred)
    y_true = np.array([np.argmax(t) if isinstance(t, np.ndarray) else t for t in y_true])  # Convert one-hot to class indices\
    y_prob = np.array(y_prob)

    # Calculate confusion matrix
    confusion_matrix_T = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix_T)

    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.5f}')

    # Calculate F1-score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1: {f1:.5f}')

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix_T.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    print(f'True Positive Rate (TPR): {tpr:.5f}')
    print(f'False Positive Rate (FPR): {fpr:.5f}')

    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    print(f'AUC: {auc:.5f}')

    # Generate a classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=6))

    # # Visualize misclassified examples
    # import random

    # # Get the indices of misclassified examples
    # misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

    # # Randomly select 5 indices
    # random_indices = random.sample(misclassified_indices, 5)

    # # Print these examples
    # for i in random_indices:
    #     print(f'True: {y_true[i]}, Predicted: {y_pred[i]}')
    #     print(f'Text: {test_df.iloc[i]["Text"]}')
    #     print('---')

    # plot out the loss on same graph, use different color for training and validation

    if mode == 1:
        import matplotlib.pyplot as plt

        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



