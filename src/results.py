from src.data import load_data, load_embeddings, load_pca, prepare_data_for_training
from src.perturbations import embed_perturbations
from src.hyperrectangles import load_hyperrectangles, print_hypercubes_statistics
from sentence_transformers import util
from tensorflow import keras
import tensorflow as tf
import pickle as pk
import pandas as pd
import numpy as np
import random
import os


def calculate_accuracy(datasets, encoding_models, batch_size):

    results_dict = {
                    'Dataset': [],
                    'Encoding Model': [],
                    'Model': [],
                    'Train Precision': [],
                    'Train Recall': [],
                    'Train F1': [],
                    'Test Precision': [],
                    'Test Recall': [],
                    'Test F1': [],
                    }

    for dataset in datasets:
        X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_data(dataset)
        for encoding_model, encoding_name in encoding_models.items():
            X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_embeddings(dataset, encoding_model, load_saved_embeddings=True, load_saved_align_mat=True, data=[X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg])
            X_train_pos, X_train_neg, X_test_pos, X_test_neg = load_pca(dataset, encoding_model, True, X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded)
            train_dataset, test_dataset = prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg, batch_size)

            tf.get_logger().setLevel('ERROR')
            directory = f'src/{dataset}/models/tf/{encoding_model}'
            for filename in sorted(os.listdir(directory)):
                f = os.path.join(directory, filename)
                if os.path.isdir(f):
                    TP_train, TN_train, FP_train, FN_train, TP_test, TN_test, FP_test, FN_test = 0, 0, 0, 0, 0, 0, 0, 0
                    model = keras.models.load_model(f)

                    # Train set accuracy
                    for x_batch_train, y_batch_train in train_dataset:
                        train_outputs = model(x_batch_train, training=False)
                        train_outputs = tf.argmax(train_outputs, axis=1)
                        y_true = np.array(y_batch_train, dtype='float64')
                        y_pred = np.array(train_outputs, dtype='float64')
                        TP_train += np.count_nonzero((y_pred - 1) * (y_true - 1))
                        TN_train += np.count_nonzero(y_pred * y_true)
                        FP_train += np.count_nonzero((y_pred - 1) * y_true)
                        FN_train += np.count_nonzero(y_pred * (y_true - 1))

                    # Test set accuracy
                    for x_batch_test, y_batch_test in test_dataset:
                        test_outputs= model(x_batch_test, training=False)
                        test_outputs = tf.argmax(test_outputs, axis=1)
                        y_true = np.array(y_batch_test, dtype='float64')
                        y_pred = np.array(test_outputs, dtype='float64')
                        TP_test += np.count_nonzero((y_pred - 1) * (y_true - 1))
                        TN_test += np.count_nonzero(y_pred * y_true)
                        FP_test += np.count_nonzero((y_pred - 1) * y_true)
                        FN_test += np.count_nonzero(y_pred * (y_true - 1))
                    

                    precision_train = TP_train / (TP_train + FP_train)
                    recall_train  = TP_train / (TP_train + FN_train)
                    f1_train  = 2 * precision_train  * recall_train  / (precision_train  + recall_train)

                    precision_test = TP_test / (TP_test + FP_test)
                    recall_test = TP_test / (TP_test + FN_test)
                    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)

                    results_dict['Dataset'].append(dataset)
                    results_dict['Encoding Model'].append(encoding_name)
                    results_dict['Model'].append(filename)
                    results_dict['Train Precision'].append(precision_train)
                    results_dict['Train Recall'].append(recall_train)
                    results_dict['Train F1'].append(f1_train)
                    results_dict['Test Precision'].append(precision_test)
                    results_dict['Test Recall'].append(recall_test)
                    results_dict['Test F1'].append(f1_test)

    
    # Calculate the mean and standard deviation for models of the same name but different seed
    results_df = pd.DataFrame(results_dict)

    # Remove the number from the networks' names
    for name in results_df['Model']:
        stripped_name = '_'.join(name.split('_')[:-1])
        results_df['Model'] = results_df['Model'].replace(name, stripped_name)

    # Get the network's names
    unique_models = results_df['Model'].unique()
    unique_encoding_models = results_df['Encoding Model'].unique()
    unique_datasets = results_df['Dataset'].unique()

    mean_std_dict = {
                    'Dataset': [],
                    'Encoding Model': [],
                    'Model': [],
                    'Train Precision': [],
                    'Train Recall': [],
                    'Train F1': [],
                    'Test Precision': [],
                    'Test Recall': [],
                    'Test F1': [],
                    }

    # Calculate mean and std
    for dataset in unique_datasets:
        for encoding_model in unique_encoding_models:
            for model in unique_models:
                model_df = results_df.loc[(results_df['Dataset'] == dataset) & (results_df['Encoding Model'] == encoding_model) & (results_df['Model'] == model)]

                precision_train = model_df['Train Precision']
                recall_train  = model_df['Train Recall']
                f1_train  = model_df['Train F1']

                precision_test = model_df['Test Precision']
                recall_test = model_df['Test Recall']
                f1_test = model_df['Test F1']
                
                mean_std_dict['Dataset'].append(dataset)
                mean_std_dict['Encoding Model'].append(encoding_model)
                mean_std_dict['Model'].append(model)
                mean_std_dict['Train Precision'].append(f'{np.mean(precision_train):.4f} ± {np.std(precision_train):.4f}')
                mean_std_dict['Train Recall'].append(f'{np.mean(recall_train):.4f} ± {np.std(recall_train):.4f}')
                mean_std_dict['Train F1'].append(f'{np.mean(f1_train):.4f} ± {np.std(f1_train):.4f}')
                mean_std_dict['Test Precision'].append(f'{np.mean(precision_test):.4f} ± {np.std(precision_test):.4f}')
                mean_std_dict['Test Recall'].append(f'{np.mean(recall_test):.4f} ± {np.std(recall_test):.4f}')
                mean_std_dict['Test F1'].append(f'{np.mean(f1_test):.4f} ± {np.std(f1_test):.4f}')

    path = f'results/'
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame(mean_std_dict).to_csv(f'{path}/results_accuracy.csv', index=False)
    return


def calculate_perturbations_accuracy(datasets, encoding_models, perturbations, batch_size):

    results_dict = {
                    'Dataset': [],
                    'Encoding Model': [],
                    'Model': [],
                    'Train Precision': [],
                    'Train Recall': [],
                    'Train F1': [],
                    'Test Precision': [],
                    'Test Recall': [],
                    'Test F1': [],
                    }

    for dataset in datasets:
        for encoding_model, encoding_name in encoding_models.items():
            X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = [], [], [], [], [], [], [], []
            for perturbation in perturbations:
                if X_train_pos_embedded == []:
                    X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations=True)
                    X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded = load_pca(dataset, encoding_model, True, X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded)
                else:
                    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p, y_train_pos_p, y_train_neg_p, y_test_pos_p, y_test_neg_p = embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations=True)
                    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p = load_pca(dataset, encoding_model, True, X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p)
                    X_train_pos_embedded = np.concatenate((X_train_pos_embedded, X_train_pos_embedded_p))
                    X_train_neg_embedded = np.concatenate((X_train_neg_embedded, X_train_neg_embedded_p))
                    X_test_pos_embedded = np.concatenate((X_test_pos_embedded, X_test_pos_embedded_p))
                    X_test_neg_embedded = np.concatenate((X_test_neg_embedded, X_test_neg_embedded_p))
                    y_train_pos = np.concatenate((y_train_pos, y_train_pos_p))
                    y_train_neg = np.concatenate((y_train_neg, y_train_neg_p))
                    y_test_pos = np.concatenate((y_test_pos, y_test_pos_p))
                    y_test_neg = np.concatenate((y_test_neg, y_test_neg_p))

            train_dataset, test_dataset = prepare_data_for_training(X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg, batch_size)

            tf.get_logger().setLevel('ERROR')
            directory = f'src/{dataset}/models/tf/{encoding_model}'
            for filename in sorted(os.listdir(directory)):
                f = os.path.join(directory, filename)
                if os.path.isdir(f):
                    TP_train, TN_train, FP_train, FN_train, TP_test, TN_test, FP_test, FN_test = 0, 0, 0, 0, 0, 0, 0, 0
                    model = keras.models.load_model(f)

                    # Train set accuracy
                    for x_batch_train, y_batch_train in train_dataset:
                        train_outputs = model(x_batch_train, training=False)
                        train_outputs = tf.argmax(train_outputs, axis=1)
                        y_true = np.array(y_batch_train, dtype='float64')
                        y_pred = np.array(train_outputs, dtype='float64')
                        TP_train += np.count_nonzero((y_pred - 1) * (y_true - 1))
                        TN_train += np.count_nonzero(y_pred * y_true)
                        FP_train += np.count_nonzero((y_pred - 1) * y_true)
                        FN_train += np.count_nonzero(y_pred * (y_true - 1))

                    # Test set accuracy
                    for x_batch_test, y_batch_test in test_dataset:
                        test_outputs= model(x_batch_test, training=False)
                        test_outputs = tf.argmax(test_outputs, axis=1)
                        y_true = np.array(y_batch_test, dtype='float64')
                        y_pred = np.array(test_outputs, dtype='float64')
                        TP_test += np.count_nonzero((y_pred - 1) * (y_true - 1))
                        TN_test += np.count_nonzero(y_pred * y_true)
                        FP_test += np.count_nonzero((y_pred - 1) * y_true)
                        FN_test += np.count_nonzero(y_pred * (y_true - 1))
                    

                    precision_train = TP_train / (TP_train + FP_train)
                    recall_train  = TP_train / (TP_train + FN_train)
                    f1_train  = 2 * precision_train  * recall_train  / (precision_train  + recall_train)

                    precision_test = TP_test / (TP_test + FP_test)
                    recall_test = TP_test / (TP_test + FN_test)
                    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)

                    results_dict['Dataset'].append(dataset)
                    results_dict['Encoding Model'].append(encoding_name)
                    results_dict['Model'].append(filename)
                    results_dict['Train Precision'].append(precision_train)
                    results_dict['Train Recall'].append(recall_train)
                    results_dict['Train F1'].append(f1_train)
                    results_dict['Test Precision'].append(precision_test)
                    results_dict['Test Recall'].append(recall_test)
                    results_dict['Test F1'].append(f1_test)

    
    # Calculate the mean and standard deviation for models of the same name but different seed
    results_df = pd.DataFrame(results_dict)

    # Remove the number from the networks' names
    for name in results_df['Model']:
        stripped_name = '_'.join(name.split('_')[:-1])
        results_df['Model'] = results_df['Model'].replace(name, stripped_name)

    # Get the network's names
    unique_models = results_df['Model'].unique()
    unique_encoding_models = results_df['Encoding Model'].unique()
    unique_datasets = results_df['Dataset'].unique()

    mean_std_dict = {
                    'Dataset': [],
                    'Encoding Model': [],
                    'Model': [],
                    'Train Precision': [],
                    'Train Recall': [],
                    'Train F1': [],
                    'Test Precision': [],
                    'Test Recall': [],
                    'Test F1': [],
                    }

    # Calculate mean and std
    for dataset in unique_datasets:
        for encoding_model in unique_encoding_models:
            for model in unique_models:
                model_df = results_df.loc[(results_df['Dataset'] == dataset) & (results_df['Encoding Model'] == encoding_model) & (results_df['Model'] == model)]

                precision_train = model_df['Train Precision']
                recall_train  = model_df['Train Recall']
                f1_train  = model_df['Train F1']

                precision_test = model_df['Test Precision']
                recall_test = model_df['Test Recall']
                f1_test = model_df['Test F1']
                
                mean_std_dict['Dataset'].append(dataset)
                mean_std_dict['Encoding Model'].append(encoding_model)
                mean_std_dict['Model'].append(model)
                mean_std_dict['Train Precision'].append(f'{np.mean(precision_train):.4f} ± {np.std(precision_train):.4f}')
                mean_std_dict['Train Recall'].append(f'{np.mean(recall_train):.4f} ± {np.std(recall_train):.4f}')
                mean_std_dict['Train F1'].append(f'{np.mean(f1_train):.4f} ± {np.std(f1_train):.4f}')
                mean_std_dict['Test Precision'].append(f'{np.mean(precision_test):.4f} ± {np.std(precision_test):.4f}')
                mean_std_dict['Test Recall'].append(f'{np.mean(recall_test):.4f} ± {np.std(recall_test):.4f}')
                mean_std_dict['Test F1'].append(f'{np.mean(f1_test):.4f} ± {np.std(f1_test):.4f}')

    path = f'results/'
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame(mean_std_dict).to_csv(f'{path}/results_perturbations_accuracy.csv', index=False)
    return


def calculate_marabou_results(datasets, encoding_models):
    marabou_dict = {
        'Dataset': [],
        'Encoding Model': [],
        'Model': [],
        'Hypercube': [],
        'is_UNSAT': [],
        'Rectangle Number': [],
    }

    for dataset in datasets:
        for encoding_model, encoding_name in encoding_models.items():
            input_path = f'verification/marabou/outputs/{encoding_name}/{dataset}/'

            for batch_folder in os.listdir(input_path):
                batch_path = f'{input_path}/{batch_folder}'

                if os.path.isdir(batch_path):
                    for filename in os.listdir(batch_path):
                        file_path = f'{input_path}/{batch_folder}/{filename}'

                        with open(file_path, "r") as f:
                            name = ""
                            model = ""
                            hypercube = ""
                            rectangle_number = None
                            is_UNSAT = "unknown"
                            exists = True
                            file_lines = f.readlines()

                            for line in file_lines:
                                if "unsat" == line.strip():
                                    is_UNSAT = "unsat"
                                elif "sat" in line and "unsat" not in line:
                                    is_UNSAT = "sat"
                                elif "doesn't exist" in line:
                                    exists = False
                                elif "TIME LIMIT" in line:
                                    is_UNSAT = "unknown"
                                elif "ipq" in line:
                                    name = f"{dataset}@" + line.split('/')[-2].replace('-','@') + "@" + line.split('/')[-1][:-5]
                                    model = f"{'_'.join(line.split('/')[-2].split('@')[0].split('_')[:-1])}"
                                    hypercube = f"{line.split('/')[-2].split('@')[-1]}"
                                    rectangle_number = f"{line.split('/')[-1].split('.')[0]}"

                            if len(file_lines) > 1 and len(name.strip()) and (model == 'base' or model == 'perturbations') and (hypercube == 'eps_cube' or hypercube == 'perturbations'):
                                marabou_dict['Dataset'].append(dataset)
                                marabou_dict['Encoding Model'].append(encoding_name)
                                marabou_dict['Model'].append(model)
                                marabou_dict['Hypercube'].append(hypercube)
                                marabou_dict['is_UNSAT'].append(is_UNSAT)
                                marabou_dict['Rectangle Number'].append(int(rectangle_number))
                    
    marabou_df = pd.DataFrame(marabou_dict)

    unique_models = marabou_df['Model'].unique()
    unique_encoding_models = marabou_df['Encoding Model'].unique()
    unique_datasets = marabou_df['Dataset'].unique()
    unique_hypercubes = marabou_df['Hypercube'].unique()

    marabou_mean_dict = {
        'Dataset': [],
        'Encoding Model': [],
        'Model': [],
        'Hypercube': [],
        'Percentage': [],
        'Number': [],
        'Total': [],
    }

    verified_hyperrectangles_dict = {
        'Dataset': [],
        'Encoding Model': [],
        'Model': [],
        'Hypercube': [],
        'Rectangles List': [],
    }

    indices_path = f'verification/marabou/indices/'
    if not os.path.exists(indices_path):
        os.makedirs(indices_path)

    for dataset in unique_datasets:
        for encoding_model in unique_encoding_models:
            for model in unique_models:
                for hypercube in unique_hypercubes:
                    is_UNSAT_df = marabou_df.loc[(marabou_df['Model'] == model) & (marabou_df['Dataset'] == dataset) & (marabou_df['Encoding Model'] == encoding_model) & (marabou_df['Hypercube'] == hypercube)]
                    try:
                        number = is_UNSAT_df['is_UNSAT'].value_counts()['unsat']
                    except:
                        number = 0
                    total = len(is_UNSAT_df.index)
                    if total == 0:
                        total = 1
                    percentage = (number / total) * 100

                    marabou_mean_dict['Model'].append(model)
                    marabou_mean_dict['Dataset'].append(dataset)
                    marabou_mean_dict['Hypercube'].append(hypercube)
                    marabou_mean_dict['Encoding Model'].append(encoding_model)
                    marabou_mean_dict['Number'].append(number)
                    marabou_mean_dict['Total'].append(total)
                    marabou_mean_dict['Percentage'].append(f'{percentage:.2f}')

                    verified_df = is_UNSAT_df.loc[(is_UNSAT_df['is_UNSAT'] == 'unsat')]

                    # verified_hyperrectangles_dict['Dataset'].append(dataset)
                    # verified_hyperrectangles_dict['Encoding Model'].append(encoding_model)
                    # verified_hyperrectangles_dict['Model'].append(model)
                    # verified_hyperrectangles_dict['Hypercube'].append(hypercube)
                    # verified_hyperrectangles_dict['Rectangles List'].append(np.array(verified_df['Rectangle Number'].unique()))
                    np.save(f'{indices_path}/{dataset}_{encoding_model}_{model}_{hypercube}.npy', verified_df['Rectangle Number'].unique())

    path = f'results/'
    if not os.path.exists(path):
        os.makedirs(path)
    marabou_results_df = pd.DataFrame(marabou_mean_dict)
    # marabou_results_df = marabou_results_df.pivot(index=['Dataset', 'Encoding Model', 'Model'], columns='Hypercube', values='Percentage').reset_index()
    marabou_results_df.to_csv(f'{path}/results_marabou.csv', index=False)

    # verified_hyperrectangles_df = pd.DataFrame(verified_hyperrectangles_dict)
    # # verified_hyperrectangles_df = verified_hyperrectangles_df.pivot(index=['Dataset', 'Encoding Model', 'Model'], columns='Hypercube', values='Rectangles List').reset_index()
    # verified_hyperrectangles_df.to_csv(f'{path}/results_verified_hyperrectangles.csv', index=False)
    
    return


def calculate_number_of_sentences_inside_the_verified_hyperrectangles(datasets, encoding_models, perturbations, h_names):
    verified_hyperrectangles_df = pd.read_csv('results/results_marabou.csv')
    unique_models = verified_hyperrectangles_df['Model'].unique()

    results_dict = {
        'Dataset': [],
        'Encoding Model': [],
        'Model': [],
        'Hypercube': [],
        'Train Pos Number': [],
        'Train Neg Number': [],
        'Test Pos Number': [],
        'Test Neg Number': [],
        'Train Pos Total': [],
        'Train Neg Total': [],
        'Test Pos Total': [],
        'Test Neg Total': [],
        'Train Pos Percentage': [],
        'Train Neg Percentage': [],
        'Test Pos Percentage': [],
        'Test Neg Percentage': [],
    }

    for dataset in datasets:
        for encoding_model, encoding_name in encoding_models.items():
            X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded = [], [], [], []
            for perturbation in perturbations:
                if X_train_pos_embedded == []:
                    X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, _, _, _, _ = embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations=True)
                    X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded = load_pca(dataset, encoding_model, True, X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded)
                else:
                    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p, _, _, _, _ = embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations=True)
                    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p = load_pca(dataset, encoding_model, True, X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p)
                    X_train_pos_embedded = np.concatenate((X_train_pos_embedded, X_train_pos_embedded_p))
                    X_train_neg_embedded = np.concatenate((X_train_neg_embedded, X_train_neg_embedded_p))
                    X_test_pos_embedded = np.concatenate((X_test_pos_embedded, X_test_pos_embedded_p))
                    X_test_neg_embedded = np.concatenate((X_test_neg_embedded, X_test_neg_embedded_p))

            for model in unique_models:
                for h, h_name in h_names.items():
                    print(f'{dataset} {encoding_name} {model} {h}')

                    indices = np.load(f'verification/marabou/indices/{dataset}_{encoding_name}_{model}_{h}.npy')
                    hyperrectangles = load_hyperrectangles(dataset, encoding_model, h_name, load_saved_hyperrectangles=True)
                    hyperrectangles = np.take(hyperrectangles, indices, axis=0)
                    
                    train_pos_percentage, test_pos_percentage, train_neg_percentage, test_neg_percentage, train_pos_n, test_pos_n, train_neg_n, test_neg_n = print_hypercubes_statistics(hyperrectangles, X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded)

                    results_dict['Dataset'].append(dataset)
                    results_dict['Encoding Model'].append(encoding_name)
                    results_dict['Model'].append(model)
                    results_dict['Hypercube'].append(h)
                    results_dict['Train Pos Number'].append(train_pos_n)
                    results_dict['Train Neg Number'].append(train_neg_n)
                    results_dict['Test Pos Number'].append(test_pos_n)
                    results_dict['Test Neg Number'].append(test_neg_n)
                    results_dict['Train Pos Total'].append(len(X_train_pos_embedded))
                    results_dict['Train Neg Total'].append(len(X_train_neg_embedded))
                    results_dict['Test Pos Total'].append(len(X_test_pos_embedded))
                    results_dict['Test Neg Total'].append(len(X_test_neg_embedded))
                    results_dict['Train Pos Percentage'].append(f'{train_pos_percentage:.2f}')
                    results_dict['Train Neg Percentage'].append(f'{train_neg_percentage:.2f}')
                    results_dict['Test Pos Percentage'].append(f'{test_pos_percentage:.2f}')
                    results_dict['Test Neg Percentage'].append(f'{test_neg_percentage:.2f}')

    path = f'results/'
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame(results_dict).to_csv(f'{path}/number_of_points_inside_verified_hyperrectangles.csv', index=False)
    return


def calculate_cosine_perturbations_filtering(datasets, encoding_models, perturbations):
    cosine_dict = {
        'Dataset': [],
        'Encoding Model': [],
        'Perturbation': [],
        'Positive': [],
        'Negative': [],
    }

    for dataset in datasets:
        X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, _, _, _, _ = load_data(dataset)

        for encoding_model, encoding_name in encoding_models.items():
            with open(f'src/{dataset}/embeddings/{encoding_model}/pca.pkl', 'rb') as pickle_file:
                data_pca = pk.load(pickle_file)

            X_train_pos, X_train_neg, _, _, _, _, _, _ = load_embeddings(dataset, encoding_model, load_saved_embeddings=True)

            for perturbation in perturbations:
                X_train_pos_p, X_train_neg_p, _, _, _, _, _, _ = embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations=True)
                train_pos_indexes = np.load(f'src/{dataset}/perturbations/{perturbation}/indexes/train_pos_indexes.npy')
                train_neg_indexes = np.load(f'src/{dataset}/perturbations/{perturbation}/indexes/train_neg_indexes.npy')
                
                p_total = len(X_train_pos_p)
                n_total = len(X_train_neg_p)
                p = 0
                n = 0
                print(perturbation)

                for i in range(len(X_train_pos)):
                    for index, value in enumerate(train_pos_indexes):
                        if value == i:
                            try:
                                cosine_score = util.cos_sim(X_train_pos[i], X_train_pos_p[index])
                                if cosine_score > 0.6:
                                    p += 1
                            except:
                                pass

                for i in range(len(X_train_neg)):
                    for index, value in enumerate(train_neg_indexes):
                        if value == i:
                            try:
                                cosine_score = util.cos_sim(X_train_neg[i], X_train_neg_p[index])
                                if cosine_score > 0.6:
                                    n += 1
                            except:
                                pass

                p_percentage = float(p)/float(p_total) * 100
                n_percentage = float(n)/float(n_total) * 100
                
                cosine_dict['Dataset'].append(dataset)
                cosine_dict['Encoding Model'].append(encoding_name)
                cosine_dict['Perturbation'].append(perturbation)
                cosine_dict['Positive'].append(f'{p}/{p_total} ({p_percentage:.2f}\%)')
                cosine_dict['Negative'].append(f'{n}/{n_total} ({n_percentage:.2f}\%)')

    path = f'results/'
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame(cosine_dict).to_csv(f'{path}/results_cosine_perturbations_filtering.csv', index=False)
    return
