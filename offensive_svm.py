#!/usr/bin/env python
'''
Train and evaluate a Support Vector Machine (SVM) for offensive language detection.
'''
import argparse
import pickle
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def create_arg_parser() -> argparse.Namespace:
    """Create the argument parser for the program."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # --- Subparser for training ---
    parser_train = subparsers.add_parser('train', help='Train and evaluate the model')
    parser_train.add_argument('train_file', type=str, help='Path to the training data (.tsv)')
    parser_train.add_argument('dev_file', type=str, help='Path to the development data (.tsv)')
    parser_train.add_argument('--save-model', type=str, help='Save the trained model to a file')
    parser_train.add_argument('--grid-search', action='store_true', help='Optimize parameters using grid search')
    parser_train.add_argument('--ngram', type=int, default=1, help='Use n-grams as features (default: 1)')
    parser_train.add_argument('--C', type=float, default=1.0, help='Regularization parameter C for LinearSVC')
    parser_train.add_argument('--max-iter', type=int, default=1000, help='Max iterations for LinearSVC')

    # --- Subparser for evaluation ---
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a saved model on the test set')
    evaluate_parser.add_argument('model_file', type=str, help='Path to the saved model file (.pkl)')
    evaluate_parser.add_argument('test_file', type=str, help='Path to the test data (.tsv)')

    return parser.parse_args()


def read_corpus(corpus_file: str) -> tuple[list[str], list[str]]:
    """Reads a .tsv corpus file and returns documents and labels."""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) == 2:
                documents.append(tokens[0])
                labels.append(tokens[1])
    return documents, labels


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train and evaluate the SVM model."""
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Create the pipeline: TF-IDF Vectorizer -> LinearSVC
    pipeline = Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1, args.ngram))),
        ('cls', LinearSVC(C=args.C, max_iter=args.max_iter, dual=True)),
    ])

    if args.grid_search:
        print("--- Starting Grid Search ---")
        # Define a focused parameter grid for LinearSVC
        param_grid = {
            'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'cls__C': [0.1, 1, 10],
            'cls__max_iter': [1000, 2000],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, Y_train)
        print(f"\nBest parameters found: {grid_search.best_params_}")
        classifier = grid_search.best_estimator_
    else:
        print("--- Training with specified parameters ---")
        classifier = pipeline
        classifier.fit(X_train, Y_train)

    # --- Evaluation ---
    Y_pred = classifier.predict(X_dev)
    print("\n Evaluation on Development Set")
    print(f"Accuracy: {accuracy_score(Y_dev, Y_pred):.4f}")
    print(f"F1-score (OFF): {f1_score(Y_dev, Y_pred, pos_label='OFF'):.4f}")
    print(f"F1-score (Macro): {f1_score(Y_dev, Y_pred, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(Y_dev, Y_pred, target_names=['NOT', 'OFF']))

    # --- Save Model ---
    if args.save_model:
        print(f"\nSaving model to {args.save_model}")
        with open(args.save_model, 'wb') as f:
            pickle.dump(classifier, f)


def evaluate_model(args: argparse.Namespace) -> None:
    """Evaluate a saved model on the test set."""
    print(f"Loading model from {args.model_file} ")
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    X_test, Y_test = read_corpus(args.test_file)
    Y_pred = model.predict(X_test)

    print("\n Evaluation on Test Set")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
    print(f"F1-score (OFF): {f1_score(Y_test, Y_pred, pos_label='OFF'):.4f}")
    print(f"F1-score (Macro): {f1_score(Y_test, Y_pred, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, target_names=['NOT', 'OFF']))


def main() -> int:
    """Main function to run the program."""
    args = create_arg_parser()
    mode_to_func: dict[str, Callable] = {
        'train': train_and_evaluate,
        'evaluate': evaluate_model,
    }
    if args.command in mode_to_func:
        mode_to_func[args.command](args)
        return 0
    else:
        print('No valid command given. Use --help for more information.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())