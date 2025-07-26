import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. Load and Label
df_fake = pd.read_csv("../data/Fake.csv")
df_real = pd.read_csv("../data/True.csv")

df_fake['label'] = 0
df_real['label'] = 1

df = pd.concat([df_fake, df_real])
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

# 2. Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 3. Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.1
)

# 4. Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# 5. Custom Dataset
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

print(train_dataset[0])