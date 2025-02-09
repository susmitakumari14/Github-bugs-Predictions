import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from transformers import BertTokenizer
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeBugDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BugClassifier(nn.Module):
    def __init__(self, n_classes, vocab_size=30522, embedding_dim=768):
        super(BugClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer with smaller hidden size
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Output layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)  # 256 = 2 * hidden_size (bidirectional)
    
    def forward(self, input_ids, attention_mask):
        # Embedding layer
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Apply BiLSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.size())
        lstm_out = lstm_out * mask_expanded
        
        # Get sequence lengths with clamping to ensure valid indices
        seq_lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        batch_size = input_ids.size(0)
        
        # Get the last valid output for each sequence
        final_output = lstm_out[
            torch.arange(batch_size, device=lstm_out.device),
            seq_lengths
        ]
        
        # Apply dropout and final classification
        out = self.dropout(final_output)
        out = self.fc(out)
        
        return out

class BugPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # Import setup to ensure NLTK data is available
        try:
            import setup
        except ImportError:
            logger.warning("setup.py not found, NLTK data may not be available")
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK resources initialized successfully")
        except LookupError as e:
            logger.error(f"NLTK resource initialization failed: {str(e)}")
            raise

    def preprocess_code(self, code):
        """Preprocess code snippet with code-specific rules."""
        try:
            # Normalize whitespace
            code = re.sub(r'\s+', ' ', code)
            # Add spaces around operators
            code = re.sub(r'([=\+\-\*/\(\)\[\]{},])', r' \1 ', code)
            # Handle special Python keywords
            code = re.sub(r'\b(def|class|if|else|while|for|in|return|print)\b', r' \1 ', code)
            return code.strip()
        except Exception as e:
            logger.error(f"Error in code preprocessing: {str(e)}")
            return str(code)

    def preprocess_text(self, text):
        try:
            if '[SEP]' in text:
                # Split code and description
                code, desc = text.split('[SEP]')
                # Process code differently from natural language
                code = self.preprocess_code(code)
                # Process description with NLTK
                desc_tokens = word_tokenize(str(desc).lower())
                desc_tokens = [self.lemmatizer.lemmatize(token) 
                             for token in desc_tokens 
                             if token not in self.stop_words]
                return code + ' [SEP] ' + ' '.join(desc_tokens)
            else:
                # If no separator, treat as code
                return self.preprocess_code(text)
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return str(text)

    def prepare_data(self, df):
        try:
            # Combine code and description for better context
            df['combined_text'] = df['code_snippet'] + ' [SEP] ' + df['bug_description']
            df['combined_text'] = df['combined_text'].apply(self.preprocess_text)
            
            # Encode labels
            self.label_encoder.fit(df['bug_type'])
            labels = self.label_encoder.transform(df['bug_type'])
            
            return df['combined_text'].values, labels
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train(self, train_data, epochs=10, batch_size=16, learning_rate=2e-5):
        texts, labels = self.prepare_data(train_data)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = CodeBugDataset(X_train, y_train, self.tokenizer)
        val_dataset = CodeBugDataset(X_val, y_val, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = BugClassifier(len(self.label_encoder.classes_))
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            logger.info(f'Epoch [{epoch+1}/{epochs}]')
            logger.info(f'Average Training Loss: {avg_train_loss:.4f}')
            logger.info(f'Training Accuracy: {train_accuracy:.2f}%')
            logger.info(f'Average Validation Loss: {avg_val_loss:.4f}')
            logger.info(f'Validation Accuracy: {accuracy:.2f}%')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info('Model saved!')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after epoch {epoch+1}')
                break

    def predict(self, code_snippet, description):
        if self.model is None:
            raise ValueError("Model needs to be trained first!")
        
        self.model.eval()
        combined_text = f"{code_snippet} [SEP] {description}"
        processed_text = self.preprocess_text(combined_text)
        
        # Prepare input
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction_prob, predicted = torch.max(probabilities, 1)
            
        predicted_label = self.label_encoder.inverse_transform([predicted.item()])[0]
        confidence = prediction_prob.item()
        
        return predicted_label, confidence

def main():
    try:
        # Load the dataset
        df = pd.read_csv('python_bug_dataset.csv')
        
        # Initialize predictor
        predictor = BugPredictor()
        
        # Train the model
        logger.info("Starting training...")
        predictor.train(df, epochs=20, batch_size=8)  # Smaller batch size for better stability
        logger.info("Training completed!")
        
        # Example predictions with confidence scores
        test_cases = [
            {
                'code': 'print "Hello World"',
                'description': 'Print statement without parentheses'
            },
            {
                'code': 'x = [1,2,3]\nprint(x[10])',
                'description': 'Accessing list beyond its length'
            },
            {
                'code': 'def func(a,b=1,c):\n    return a+b+c',
                'description': 'Function parameter order issue'
            }
        ]
        
        print("\nTesting model predictions:\n")
        for case in test_cases:
            predicted_type, confidence = predictor.predict(case['code'], case['description'])
            print(f"Code:\n{case['code']}")
            print(f"Description: {case['description']}")
            print(f"Predicted Bug Type: {predicted_type}")
            print(f"Confidence: {confidence:.2%}\n")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
