from src.data import load_embeddings
from src.perturbations import embed_perturbations
from sentence_transformers import util
import pickle as pk
import numpy as np
import os


def contained(point, hypercube):
    for i in range(len(point)):
        if point[i] < hypercube[i][0] or point[i] > hypercube[i][1]:
            return False
    return True


def calculate_hypercube(points):
    # Calculate the hypercube around the points
    min_max_list = np.full((points.shape[1], 2), [np.inf, -np.inf])
    for point in points:
        for i in range(len(point)):
            if point[i] < min_max_list[i][0]:
                min_max_list[i][0] = point[i]
            if point[i] > min_max_list[i][1]:
                min_max_list[i][1] = point[i]
    return min_max_list


def print_hypercubes_statistics(hypercubes, X_train_pos, X_test_pos, X_train_neg, X_test_neg):
    train_pos_percentage, test_pos_percentage, train_neg_percentage, test_neg_percentage, train_pos_n, test_pos_n, train_neg_n, test_neg_n = 0, 0, 0, 0, 0, 0, 0, 0
    # Check how many points are contained in the union of the hypercubes
    # Training positive points
    if np.any(X_train_pos):
        t1 = 0
        for p in range(len(X_train_pos)):
            for h in range(len(hypercubes)):
                if contained(X_train_pos[p], hypercubes[h]):
                    t1+=1
                    # print(p)
                    break
        f1 = len(X_train_pos) - t1
        train_pos_percentage = float(t1)/float(t1+f1) * 100
        train_pos_n = t1
        print(f' Train positive points inside the hypercubes: {train_pos_percentage:.2f}% (T:{t1} - F:{f1})')

    # Testing positive points
    if np.any(X_test_pos):
        t2 = 0
        for p in range(len(X_test_pos)):
            for h in range(len(hypercubes)):
                if contained(X_test_pos[p], hypercubes[h]):
                    t2+=1
                    # print(p)
                    break
        f2 = len(X_test_pos) - t2
        test_pos_percentage = float(t2)/float(t2+f2) * 100
        test_pos_n = t2
        print(f' Test positive points inside the hypercubes: {test_pos_percentage:.2f}% (T:{t2} - F:{f2})')

    # Training negative points
    if np.any(X_train_neg):
        t3 = 0
        for p in range(len(X_train_neg)):
            for h in range(len(hypercubes)):
                if contained(X_train_neg[p], hypercubes[h]):
                    t3+=1
                    break
        f3 = len(X_train_neg) - t3
        train_neg_percentage = float(t3)/float(t3+f3) * 100
        train_neg_n = t3
        print(f' Train negative points inside the hypercubes: {train_neg_percentage:.2f}% (T:{t3} - F:{f3})')

    # Testing negative points
    if np.any(X_test_neg):
        t4 = 0
        for p in range(len(X_test_neg)):
            for h in range(len(hypercubes)):
                if contained(X_test_neg[p], hypercubes[h]):
                    t4+=1
                    break
        f4 = len(X_test_neg) - t4
        test_neg_percentage = float(t4)/float(t4+f4) * 100
        test_neg_n = t4
        print(f' Test negative points inside the hypercubes: {test_neg_percentage:.2f}% (T:{t4} - F:{f4})')
    
    return train_pos_percentage, test_pos_percentage, train_neg_percentage, test_neg_percentage, train_pos_n, test_pos_n, train_neg_n, test_neg_n


def load_hyperrectangles(dataset, encoding_model, h_name, load_saved_hyperrectangles, eps=0.05):
    if load_saved_hyperrectangles:
        hyperrectangles = []
        for h_n in h_name:
            if hyperrectangles == []:
                hyperrectangles = np.load(f'src/{dataset}/hyperrectangles/{encoding_model}/{h_n}.npy')
            else:
                hyperrectangles = np.concatenate((hyperrectangles, np.load(f'src/{dataset}/hyperrectangles/{encoding_model}/{h_n}.npy')), axis=0)

    else:
        path = f'src/{dataset}/hyperrectangles/{encoding_model}'
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'src/{dataset}/embeddings/{encoding_model}/pca.pkl', 'rb') as pickle_file:
            data_pca = pk.load(pickle_file)

        X_train_pos_embedded, _, _, _, _, _, _, _ = load_embeddings(dataset, encoding_model, load_saved_embeddings=True)
        X_train_pos = data_pca.transform(X_train_pos_embedded)
        hyperrectangles = []

        if h_name == 'character' or h_name == 'word' or h_name == 'vicuna':
            train_pos_index = np.load(f'src/{dataset}/perturbations/{h_name}/indexes/train_pos_indexes.npy')
            X_train_pos_p_embedded, _, _, _, _, _, _, _ = embed_perturbations(dataset, h_name, encoding_model, load_saved_perturbations=True)
            X_train_pos_p = data_pca.transform(X_train_pos_p_embedded)

            for i in range(len(X_train_pos)):
                points = []
                points.append(X_train_pos[i])
                for index, value in enumerate(train_pos_index):
                    if value == i:
                        cosine_score = util.cos_sim(X_train_pos_embedded[i], X_train_pos_p_embedded[index])
                        if cosine_score > 0.6:
                            points.append(X_train_pos_p[index])
                if len(points) >= 2:
                    hyperrectangle = calculate_hypercube(np.array(points))
                    hyperrectangles.append(hyperrectangle)

        elif h_name == 'eps_cube':
            for p in X_train_pos:
                eps_cube = []
                for d in p:
                    eps_cube.append(np.array([d - eps, d + eps]))
                hyperrectangles.append(np.array(eps_cube))

        print(np.array(hyperrectangles).shape)
        np.save(f'{path}/{h_name}.npy', hyperrectangles)

    return hyperrectangles
