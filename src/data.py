from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import tensorflow as tf
import pickle as pk
import pandas as pd
import numpy as np
import os


def load_data(dataset):
    if dataset == 'ruarobot':
        # Load the csv files
        train_pos = pd.read_csv(f'src/{dataset}/data/pos.train.csv')
        train_neg = pd.read_csv(f'src/{dataset}/data/neg.train.csv')
        train_amb = pd.read_csv(f'src/{dataset}/data/amb.train.csv')
        val_pos = pd.read_csv(f'src/{dataset}/data/pos.val.csv')
        val_neg = pd.read_csv(f'src/{dataset}/data/neg.val.csv')
        val_amb = pd.read_csv(f'src/{dataset}/data/amb.val.csv')
        test_pos = pd.read_csv(f'src/{dataset}/data/pos.test.csv')
        test_neg = pd.read_csv(f'src/{dataset}/data/neg.test.csv')
        test_amb = pd.read_csv(f'src/{dataset}/data/amb.test.csv')

        # Concatenate training and validation data
        train_pos = pd.concat([train_pos, val_pos])
        train_neg = pd.concat([train_neg, val_neg])
        train_amb = pd.concat([train_amb, val_amb])

        # Get the sentences and labels
        X_train_pos = train_pos['text'].to_numpy()
        X_train_neg = train_neg['text'].to_numpy()
        X_train_amb = train_amb['text'].to_numpy()
        X_test_pos = test_pos['text'].to_numpy()
        X_test_neg = test_neg['text'].to_numpy()
        X_test_amb = test_amb['text'].to_numpy()

        y_train_pos = train_pos['label'].replace({'p': 0, 'n': 1, 'a': 0}).to_numpy()
        y_train_neg = train_neg['label'].replace({'p': 0, 'n': 1, 'a': 0}).to_numpy()
        y_train_amb = train_amb['label'].replace({'p': 0, 'n': 1, 'a': 0}).to_numpy()
        y_test_pos = test_pos['label'].replace({'p': 0, 'n': 1, 'a': 0}).to_numpy()
        y_test_neg = test_neg['label'].replace({'p': 0, 'n': 1, 'a': 0}).to_numpy()
        y_test_amb = test_amb['label'].replace({'p': 0, 'n': 1, 'a': 0}).to_numpy()

        # Concatenate the positive and ambiguous sentences and labels
        X_train_pos = np.concatenate((X_train_pos, X_train_amb), axis=0)
        X_test_pos = np.concatenate((X_test_pos, X_test_amb), axis=0)

        y_train_pos = np.concatenate((y_train_pos, y_train_amb), axis=0)
        y_test_pos = np.concatenate((y_test_pos, y_test_amb), axis=0)
    
    elif dataset == 'medical':
        # Load the csv file
        expert = pd.read_csv(f'src/{dataset}/data/medicheck-expert.csv')
        med_neg = pd.read_csv(f'src/{dataset}/data/medicheck-neg.csv')

        # Select the X and y columns, then the rows that belongs to classes 1 to 3, and finally replace the labes with only 1
        expert = expert[['query', 'query-label-expert']]
        pos = expert.loc[(expert['query-label-expert'] >= 1) & (expert['query-label-expert'] <= 3)]
        pos['query-label-expert'] = pos['query-label-expert'].replace({1: 0, 2: 0, 3: 0})

        neg = expert.loc[expert['query-label-expert'] == 0]
        neg['query-label-expert'] = neg['query-label-expert'].replace({0: 1})

        train_pos = pos.iloc[:(int(len(pos)*0.7))]
        test_pos = pos.iloc[(int(len(pos)*0.7)):]

        X_train_pos = train_pos['query'].to_numpy()
        X_test_pos = test_pos['query'].to_numpy()
        y_train_pos = train_pos['query-label-expert'].to_numpy()
        y_test_pos = test_pos['query-label-expert'].to_numpy()

        # Load other sentences
        neg = pd.concat([neg, med_neg])
        neg['query-label-expert'] = neg['query-label-expert'].fillna(1)

        train_neg = neg.iloc[:(int(len(neg)*0.7))]
        test_neg = neg.iloc[(int(len(neg)*0.7)):]

        X_train_neg = train_neg['query'].to_numpy()
        X_test_neg = test_neg['query'].to_numpy()
        y_train_neg = train_neg['query-label-expert'].to_numpy()
        y_test_neg = test_neg['query-label-expert'].to_numpy()

    return X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg


def load_align_mat(dataset, encoding_model, data, load_saved_align_mat):
    if load_saved_align_mat:
        align_mat = np.load(f'src/{dataset}/embeddings/{encoding_model}/align_mat.npy')

    else:
        # Rotate the data, aligning them to the axis
        print(data.shape)
        u, s, vh = np.linalg.svd(a=data)
        align_mat = np.linalg.solve(a=vh, b=np.eye(len(data[0])))

        # Save the alignment matrix
        path = f'src/{dataset}/embeddings/{encoding_model}'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f'{path}/align_mat.npy', align_mat)

    return align_mat


def load_embeddings(dataset, encoding_model, load_saved_embeddings, load_saved_align_mat=None, data=None):
    if load_saved_embeddings:
        X_train_pos = np.load(f'src/{dataset}/embeddings/{encoding_model}/X_train_pos.npy')
        X_train_neg = np.load(f'src/{dataset}/embeddings/{encoding_model}/X_train_neg.npy')
        X_test_pos = np.load(f'src/{dataset}/embeddings/{encoding_model}/X_test_pos.npy')
        X_test_neg = np.load(f'src/{dataset}/embeddings/{encoding_model}/X_test_neg.npy')
        y_train_pos = np.load(f'src/{dataset}/embeddings/{encoding_model}/y_train_pos.npy')
        y_train_neg = np.load(f'src/{dataset}/embeddings/{encoding_model}/y_train_neg.npy')
        y_test_pos = np.load(f'src/{dataset}/embeddings/{encoding_model}/y_test_pos.npy')
        y_test_neg = np.load(f'src/{dataset}/embeddings/{encoding_model}/y_test_neg.npy')

    else:
        X_train_pos = data[0]
        X_train_neg = data[1]
        X_test_pos = data[2]
        X_test_neg = data[3]
        y_train_pos = data[4]
        y_train_neg = data[5]
        y_test_pos = data[6]
        y_test_neg = data[7]

        # Embed the sentences
        encoder = SentenceTransformer(f'{encoding_model}')
        X_train_pos = encoder.encode(X_train_pos, show_progress_bar=False)
        X_train_neg = encoder.encode(X_train_neg, show_progress_bar=False)
        X_test_pos = encoder.encode(X_test_pos, show_progress_bar=False)
        X_test_neg = encoder.encode(X_test_neg, show_progress_bar=False)

        # Rotate the data
        align_mat = load_align_mat(dataset, encoding_model, X_train_pos, load_saved_align_mat)
        X_train_pos = np.matmul(X_train_pos, align_mat)
        X_train_neg = np.matmul(X_train_neg, align_mat)
        X_test_pos = np.matmul(X_test_pos, align_mat)
        X_test_neg = np.matmul(X_test_neg, align_mat)

        # Save the rotated embedded sentences and labels
        path = f'src/{dataset}/embeddings/{encoding_model}'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f'{path}/X_train_pos.npy', X_train_pos)
        np.save(f'{path}/X_train_neg.npy', X_train_neg)
        np.save(f'{path}/X_test_pos.npy', X_test_pos)
        np.save(f'{path}/X_test_neg.npy', X_test_neg)
        np.save(f'{path}/y_train_pos.npy', y_train_pos)
        np.save(f'{path}/y_train_neg.npy', y_train_neg)
        np.save(f'{path}/y_test_pos.npy', y_test_pos)
        np.save(f'{path}/y_test_neg.npy', y_test_neg)

    # # Print the shape of the rotated embedded sentences
    # print(f'Train pos sentence embeddings shape: {X_train_pos.shape}')
    # print(f'Train neg sentence embeddings shape: {X_train_neg.shape}')
    # print(f'Test pos sentence embeddings shape: {X_test_pos.shape}')
    # print(f'Test neg sentence embeddings shape: {X_test_neg.shape}')

    return X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg


def load_pca(dataset, encoding_model, load_saved_pca, X_train_pos, X_train_neg, X_test_pos, X_test_neg,  n_components=30):
    if load_saved_pca:
        with open(f'src/{dataset}/embeddings/{encoding_model}/pca.pkl', 'rb') as pickle_file:
            data_pca = pk.load(pickle_file)

    else:
        # All data:
        data = np.vstack([X_train_pos, X_train_neg, X_test_pos, X_test_neg])
        # PCA data
        data_pca = PCA(n_components=n_components).fit(data)
        # Save the PCA
        path = f'src/{dataset}/embeddings/{encoding_model}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/pca.pkl', 'wb') as pickle_file:
            pk.dump(data_pca, pickle_file)

    X_train_pos = data_pca.transform(X_train_pos)
    X_train_neg = data_pca.transform(X_train_neg)
    X_test_pos = data_pca.transform(X_test_pos)
    X_test_neg = data_pca.transform(X_test_neg)

    # # Print the shape of the PCA data
    # print(f'Train pos sentence embeddings shape: {X_train_pos.shape}')
    # print(f'Train neg sentence embeddings shape: {X_train_neg.shape}')
    # print(f'Test pos sentence embeddings shape: {X_test_pos.shape}')
    # print(f'Test neg sentence embeddings shape: {X_test_neg.shape}')

    return X_train_pos, X_train_neg, X_test_pos, X_test_neg


def prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg, batch_size):
    # Concatenate the pos and neg embeddings and labels
    X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
    X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)
    y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
    y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

    # Get the train and test datasets transformations from tensorflow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset
