#!/usr/bin/env python
'''Create and train an LSTM network for offensive language detection'''
import argparse
import random as python_random

import numpy as np
import tensorflow as tf

# Disable XLA/JIT compilation to avoid environment issues on the cluster
tf.config.optimizer.set_jit(False)

from keras.initializers import Constant
from keras.layers import Bidirectional, Dense, Embedding, LSTM
from keras.models import Sequential
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.layers import TextVectorization

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    """Creates the argument parser for command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--train_file', default='train.tsv', type=str,
        help='Input file to learn from (default train.tsv)',
    )
    parser.add_argument(
        '-d', '--dev_file', type=str, default='dev.tsv',
        help='Separate dev set to read in (default dev.tsv)',
    )
    parser.add_argument(
        '-t', '--test_file', type=str,
        help='If added, use trained model to predict on test set',
    )
    parser.add_argument(
        '-e', '--embeddings', default='glove.twitter.27B.100d.txt', type=str,
        help='Embedding file we are using (default glove.twitter.27B.100d.txt)',
    )

    parser.add_argument(
        '--learning_rate', type=float, default=0.01,
        help='Learning rate for optimizer (default 0.01)',
    )
    parser.add_argument(
        '--loss_function', type=str, default='binary_crossentropy',
        help='Loss function for training (default binary_crossentropy)',
    )
    parser.add_argument(
        '--optimizer', type=str, default='SGD',
        choices=['SGD', 'Adam', 'RMSprop'],
        help='Optimizer to use for training (default SGD)',
    )
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Verbosity mode for training (0=silent, 1=progress bar)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for training (default 16)',
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of epochs to train (default 20)',
    )

    parser.add_argument(
        '--lstm_units', type=int, default=128,
        help='Number of LSTM units (default 128)',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help='Dropout rate for LSTM (default 0.2)',
    )
    parser.add_argument(
        '--recurrent_dropout', type=float, default=0.2,
        help='Recurrent dropout rate for LSTM (default 0.2)',
    )
    parser.add_argument(
        '--lstm_layers', type=int, default=1,
        help='How many LSTM layers will be used (default 1)',
    )
    # parser.add_argument(
    #     '--bidirectional_layer', type=bool, default=False,
    #     help='Will the LSTM be bidirectional (default false)',
    # )

    parser.add_argument(
        '--bidirectional_layer', action='store_true',
        help='If specified, the LSTM will be bidirectional.',
    )

    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Read in data set and return docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            # Ensure the line has exactly two columns
            if len(tokens) == 2:
                documents.append(tokens[0])
                labels.append(tokens[1])
    return documents, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from a file and save as a dictionary.'''
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))

    embedding_dim = len(next(iter(emb.values())))

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model(emb_matrix, args):
    '''Create the Keras model to use'''
    if args.optimizer == 'SGD':
        optim = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
    elif args.optimizer == 'Adam':
        optim = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    else:
        optim = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)

    num_tokens, embedding_dim = emb_matrix.shape

    # Build the model
    model = Sequential()
    model.add(
        Embedding(
            num_tokens, embedding_dim,
            embeddings_initializer=Constant(emb_matrix),
            trainable=False,
        ),
    )

    for i in range(args.lstm_layers):
        return_sequences = i < args.lstm_layers - 1
        lstm_layer = LSTM(
            units=args.lstm_units,
            dropout=args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            return_sequences=return_sequences,
        )
        if args.bidirectional_layer:
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)

    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss=args.loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, args):
    '''Train the model'''
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(
        X_train, Y_train,
        verbose=args.verbose,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        validation_data=(X_dev, Y_dev),
    )
    test_set_predict(model, X_dev, Y_dev, 'dev')
    return model


def test_set_predict(model, X_test, Y_test, ident):
    '''Make predictions and evaluate on a given dataset'''
    Y_pred_probs = model.predict(X_test)
    Y_pred = (Y_pred_probs > 0.5).astype("int32")

    print(f'\n--- Evaluation on {ident} set ---')
    print(f'Accuracy: {accuracy_score(Y_test, Y_pred):.3f}')
    print(f'F1 Score (OFF): {f1_score(Y_test, Y_pred, pos_label=1):.3f}')
    print(f'Macro F1 Score: {f1_score(Y_test, Y_pred, average="macro"):.3f}')
    print('Classification Report:')
    print(classification_report(Y_test, Y_pred, target_names=['NOT', 'OFF'], zero_division=0))
    print('--------------------------')


def main():
    '''Main function to train and test neural network'''
    args = create_arg_parser()

    # Read data and embeddings
    X_train, Y_train_text = read_corpus(args.train_file)
    X_dev, Y_dev_text = read_corpus(args.dev_file)
    print(f"Reading embeddings from: {args.embeddings}")
    embeddings = read_embeddings(args.embeddings)

    # Vectorize text data
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Binarize labels (OFF -> 1, NOT -> 0)
    Y_train = np.array([1 if label == 'OFF' else 0 for label in Y_train_text])
    Y_dev = np.array([1 if label == 'OFF' else 0 for label in Y_dev_text])

    # Transform text to sequences of integers
    X_train_vect = vectorizer(np.array(X_train)).numpy()
    X_dev_vect = vectorizer(np.array(X_dev)).numpy()

    # Create and train the model
    model = create_model(emb_matrix, args)
    model = train_model(model, X_train_vect, Y_train, X_dev_vect, Y_dev, args)

    print("\n--- Final Run Arguments ---")
    print(args)
    print("---------------------------\n")

    # Evaluate on the test set if provided
    if args.test_file:
        X_test, Y_test_text = read_corpus(args.test_file)
        Y_test = np.array([1 if label == 'OFF' else 0 for label in Y_test_text])
        X_test_vect = vectorizer(np.array(X_test)).numpy()
        test_set_predict(model, X_test_vect, Y_test, 'test')


if __name__ == '__main__':
    main()