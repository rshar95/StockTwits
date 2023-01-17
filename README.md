# StockTwits Sentiment Analysis

This project is a deep learning model for the classification of sentiments of messages between investors and traders on StockTwits. The model uses the PyTorch library to implement a pre-trained BERT model for feature extraction and fine-tunes the last layer to classify sentiments with an accuracy of 98%. The model also generates a signal of the public sentiment for various ticker symbols.

## Requirements
- Python 3.6 or higher
- PyTorch 1.5.0 or higher
- transformers 2.11.0 or higher

## Usage
1. Clone the repository

```
sh
git clone https://github.com/[username]/stocktwits-sentiment-analysis.git

```


2. Install the required packages

```
pip install -r requirements.txt
```


3. Run the script

```
python stocktwits.py
```


## Data
The dataset used for training and testing the model is not included in this repository due to the private nature of the data. You can use your own dataset and change the data loading code accordingly.

## Model
The model is implemented in `SentimentNet` class in `main.py` file. The class utilizes the pre-trained BERT model to extract the features from the input text and fine-tune the last layer of BERT to classify the sentiment of the text.

## Training
The model is trained using the `train()` function in the `main.py` file. It takes the SentimentNet model, data loader, loss function, optimizer, and number of training epochs as inputs.

## Evaluation
The model's performance is evaluated on the test set after each training epoch. The evaluation metric used is accuracy.

## Note
The code is provided as an example and may require modifications to work with your specific dataset and problem.




