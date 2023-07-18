from src.data import load_data, load_embeddings, load_pca, prepare_data_for_training
from src.perturbations import save_perturbations, embed_perturbations
from src.hyperrectangles import load_hyperrectangles
from src.train import train_base, train_adversarial
from verification.marabou.marabou_property_parser import parse_properties
from src.results import calculate_accuracy, calculate_perturbations_accuracy, calculate_marabou_results, calculate_number_of_sentences_inside_the_verified_hyperrectangles, calculate_cosine_perturbations_filtering
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


def get_model(n_components):
    inputs = keras.Input(shape=(n_components,), name="embeddings")
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(inputs)
    outputs = keras.layers.Dense(2, activation="linear", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


if __name__ == '__main__':
    encoding_models = {'all-MiniLM-L6-v2': 'sbert22M', 'Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit': 'sgpt13B', 'Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit': 'sgpt27B'}
    perturbations = ['character', 'word', 'vicuna']
    hyperrectangles = {'eps_cube': ['eps_cube'], 'perturbations': ['character', 'word', 'vicuna']}
    datasets = ['ruarobot', 'medical']
    seeds = [0, 3, 7, 42, 88]
    load_saved_align_mat = True
    load_saved_embeddings = True
    load_saved_pca = True
    load_saved_perturbations = True
    load_saved_hyperrectangles = True
    from_logits = True
    n_components = 30
    batch_size = 64
    pgd_steps = 5
    epochs = 50
    
    for dataset in datasets:
        # Encode the datasets
        X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_data(dataset)
        for encoding_model in encoding_models:
            X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_embeddings(dataset, encoding_model, load_saved_embeddings, load_saved_align_mat, data=[X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg])
            X_train_pos, X_train_neg, X_test_pos, X_test_neg = load_pca(dataset, encoding_model, load_saved_pca, X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, n_components)
        
        # Create and embed the perturbations
        for perturbation in perturbations:
            X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_data(dataset)
            save_perturbations(dataset, perturbation, data=[X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg])
            for encoding_model in encoding_models:
                _, _, _, _, _, _, _, _ = embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations)

        # Create the hyper-rectangles
        for encoding_model, encoding_name in encoding_models.items():
            for hyperrectangle in hyperrectangles:
                _ = load_hyperrectangles(dataset, encoding_model, hyperrectangle, load_saved_hyperrectangles)

        # Train the models
        for encoding_model in encoding_models:
            path = f'src/{dataset}/models/tf/{encoding_model}'
            if not os.path.exists(path):
                os.makedirs(path)

            X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_data(dataset)
            X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg = load_embeddings(dataset, encoding_model, load_saved_embeddings, load_saved_align_mat, data=[X_train_pos_not_embedded, X_train_neg_not_embedded, X_test_pos_not_embedded, X_test_neg_not_embedded, y_train_pos, y_train_neg, y_test_pos, y_test_neg])
            X_train_pos, X_train_neg, X_test_pos, X_test_neg = load_pca(dataset, encoding_model, load_saved_pca, X_train_pos_embedded, X_train_neg_embedded, X_test_pos_embedded, X_test_neg_embedded, n_components)
            train_dataset, test_dataset = prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg, batch_size)
            

            for seed in seeds:
                tf.random.set_seed(seed)
                np.random.seed(seed)
                n_samples = int(len(X_train_pos))
                
                model = get_model(n_components)
                model = train_base(model, train_dataset, test_dataset, epochs, from_logits=from_logits)
                model.save(f'{path}/base_{seed}')

                model = get_model(n_components)
                hyperrectangles = load_hyperrectangles(dataset, encoding_model, h_name=['character', 'word', 'vicuna'], load_saved_hyperrectangles=True)
                model = train_adversarial(model, train_dataset, test_dataset, hyperrectangles, epochs, batch_size, n_samples, pgd_steps, from_logits=from_logits)
                model.save(f'{path}/perturbations_{seed}')
    
    # Marabou
    parse_properties(datasets, encoding_models, hyperrectangles)

    # Results
    calculate_accuracy(datasets, encoding_models, batch_size)
    calculate_perturbations_accuracy(datasets, encoding_models, perturbations, batch_size)
    calculate_number_of_sentences_inside_the_verified_hyperrectangles(datasets, encoding_models, perturbations, hyperrectangles)
    calculate_cosine_perturbations_filtering(datasets, encoding_models, perturbations)
    calculate_marabou_results(datasets, encoding_models)
    




















