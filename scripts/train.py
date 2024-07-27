import os
import torch
import argparse
from transformers import XLNetForSequenceClassification, XLNetTokenizer, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import get_device, load_data, tokenize_data, create_data_loader

def main(data_path):
    # Parameters
    model_name = 'xlnet-base-cased'
    max_length = 128
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5

    # Load Dataset
    texts, labels = load_data(data_path)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    # Tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(model_name)

    # Tokenize Data
    train_inputs, train_masks, train_labels = tokenize_data(train_texts, train_labels, tokenizer, max_length)
    val_inputs, val_masks, val_labels = tokenize_data(val_texts, val_labels, tokenizer, max_length)

    # DataLoader
    train_loader = create_data_loader(train_inputs, train_masks, train_labels, batch_size, RandomSampler)
    val_loader = create_data_loader(val_inputs, val_masks, val_labels, batch_size, SequentialSampler)

    # Model
    device = get_device()
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(label_encoder, os.path.join(model_dir, 'label_encoder.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the data')
    args = parser.parse_args()
    main(args.data_path)
