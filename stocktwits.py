import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

class SentimentNet(nn.Module):
    """
    SentimentNet is a PyTorch module for sentiment analysis.
    It utilizes the pre-trained BERT model to extract the features from the input text and
    fine-tune the last layer of BERT to classify the sentiment of the text.
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(SentimentNet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, x, attention_mask):
        """
        Forward pass of the SentimentNet model.
        Inputs:
            x: input text, shape (batch_size, seq_len)
            attention_mask: attention mask for input text, shape (batch_size, seq_len)
        Outputs:
            logits: output logits of the model, shape (batch_size, output_dim)
        """
        _, x = self.bert(x, attention_mask=attention_mask)
        x = self.fc(x[-1])
        return x
    
def train(model, dataloader, criterion, optimizer, num_epochs):
    """
    Train the SentimentNet model.
    Inputs:
        model: SentimentNet model
        dataloader: data loader for the training dataset
        criterion: loss function
        optimizer: optimizer for the model's parameters
        num_epochs: number of training epochs
    """
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data, attention_masks)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
        # evaluate the model
        correct = 0
        total = 0
        for data, labels in dataloader:
            output = model(data, attention_masks)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        accuracy = 100 * correct / total
        print('Epoch:', epoch+1, 'Accuracy:', accuracy)

if __name__ == '__main__':
    # Load the pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Convert the text to input ids and attention masks
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    attention_masks = [1] * len(input_ids)

    # Padding
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_masks = attention_mas
