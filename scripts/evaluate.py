import torch
import argparse
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from sklearn.metrics import accuracy_score, f1_score
from utils import get_device, load_data, tokenize_data, create_data_loader
import os

def main(data_path):
    # Parameters
    model_dir = './models'
    max_length = 128
    batch_size = 16

    # Load Model, Tokenizer, and Label Encoder
    model = XLNetForSequenceClassification.from_pretrained(model_dir)
    tokenizer = XLNetTokenizer.from_pretrained(model_dir)
    label_encoder = torch.load(os.path.join(model_dir, 'label_encoder.pth'))

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    texts, labels = load_data(data_path)
    labels = label_encoder.transform(labels)

    # Tokenize Data
    inputs, masks, labels = tokenize_data(texts, labels, tokenizer, max_length)

    # DataLoader
    test_loader = create_data_loader(inputs, masks, labels, batch_size, SequentialSampler)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_preds = []
        total_labels = []

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids = batch[0].to(device)
                b_attention_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                outputs = model(b_input_ids, attention_mask=b_attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                total_preds.extend(preds.cpu().numpy())
                total_labels.extend(b_labels.cpu().numpy())

        accuracy = accuracy_score(total_labels, total_preds)
        f1 = f1_score(total_labels, total_preds, average='weighted')

        return accuracy, f1

    # Evaluate
    accuracy, f1 = evaluate(model, test_loader, device)
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    args = parser.parse_args()
    main(args.data_path)
