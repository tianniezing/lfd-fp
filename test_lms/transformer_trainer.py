#!/usr/bin/env python
'''
Finetunes a Transformer model for offensive language detection.
'''
import argparse
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback

def create_arg_parser():
    """Creates the argument parser for command line arguments."""
    p = argparse.ArgumentParser(description='Fine-tune a Transformer for text classification.')
    p.add_argument('--model_name', default='bert-base-uncased', help='Name of the Hugging Face model to use.')
    p.add_argument('--train_file', required=True, help='Path to the training data (.tsv file).')
    p.add_argument('--dev_file', required=True, help='Path to the development data (.tsv file).')
    p.add_argument('--test_file', default=None, help='Path to the test data (.tsv file).')
    p.add_argument('--output_dir', default='./out', help='Directory to save training outputs.')
    p.add_argument('--num_train_epochs', type=int, default=15, help='Number of training epochs.')
    p.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer.')
    p.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size for training.')
    p.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Batch size for evaluation.')
    p.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenizer.')
    p.add_argument('--early_stopping_patience', type=int, default=3, help='Patience for early stopping.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return p.parse_args()

def read_corpus(corpus_file):
    """Reads a two-column TSV file (text, label) for the offensive language task."""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                documents.append(parts[0])
                labels.append(parts[1])
    return documents, labels

def compute_metrics(eval_pred):
    """Computes accuracy and F1 scores for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': accuracy, 'f1_macro': f1}

def main():
    """Main function to orchestrate training and evaluation."""
    args = create_arg_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(Y_train)
    y_dev_enc = le.transform(Y_dev)

    tok_train = tokenizer(X_train, padding=True, truncation=True, max_length=args.max_length)
    tok_dev = tokenizer(X_dev, padding=True, truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_dict({'labels': y_train_enc, **tok_train}).with_format('torch')
    dev_ds = Dataset.from_dict({'labels': y_dev_enc, **tok_dev}).with_format('torch')

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(le.classes_)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="epoch",
        logging_strategy="epoch",
        seed=args.seed,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="eval_f1_macro",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    print("Starting Training")
    trainer.train()

    print("\n Evaluating on Development Set")
    dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
    print(f"Dev Accuracy: {dev_metrics['eval_accuracy']:.4f}")
    print(f"Dev F1-Macro: {dev_metrics['eval_f1_macro']:.4f}")

    if args.test_file:
        print("\n Evaluating on Test Set")
        X_test, Y_test = read_corpus(args.test_file)
        y_test_enc = le.transform(Y_test)
        tok_test = tokenizer(X_test, padding=True, truncation=True, max_length=args.max_length)
        test_ds = Dataset.from_dict({'labels': y_test_enc, **tok_test}).with_format('torch')
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
        print(f"Test F1-Macro: {test_metrics['eval_f1_macro']:.4f}")

if __name__ == '__main__':
    main()