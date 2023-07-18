from src.data import load_align_mat
from sentence_transformers import SentenceTransformer
from nltk.stem.wordnet import WordNetLemmatizer
from random import randint, seed
import pandas as pd
import numpy as np
import replicate
import mlconjug3
import time
import nltk
import csv
import re
import os


def save_perturbations(dataset, perturbation, data):
    if perturbation == 'character':
        perturbations = [char_swapping, char_replacement, char_deletion, char_insertion, char_repetition]
    elif perturbation == 'word':
        perturbations = [word_deletion, word_repetition, word_negation, word_ordering, word_singular_plural_verb, word_verb_tense]
    seed(42)

    path_sentences = f'src/{dataset}/perturbations/{perturbation}/sentences'
    if not os.path.exists(path_sentences):
        os.makedirs(path_sentences)
        
    path_indexes = f'src/{dataset}/perturbations/{perturbation}/indexes'
    if not os.path.exists(path_indexes):
        os.makedirs(path_indexes)

    if perturbation == 'character' or perturbation == 'word':
        # Train positive
        X_train_pos_perturbed = []
        y_train_pos_perturbed = []
        train_pos_index = []
        for i in range(len(data[0])):
            for perturbation in perturbations:
                p_perturbed = perturbation([data[0][i]])
                X_train_pos_perturbed.append(p_perturbed[0])
                y_train_pos_perturbed.append(data[4][i])
                train_pos_index.append(i)
        np.save(f'{path_sentences}/X_train_pos.npy', X_train_pos_perturbed)
        np.save(f'{path_sentences}/y_train_pos.npy', y_train_pos_perturbed)
        np.save(f'{path_indexes}/train_pos_indexes.npy', train_pos_index)
        
        # Train negative
        X_train_neg_perturbed = []
        y_train_neg_perturbed = []
        train_neg_index = []
        for i in range(len(data[1])):
            for perturbation in perturbations:
                p_perturbed = perturbation([data[1][i]])
                X_train_neg_perturbed.append(p_perturbed[0])
                y_train_neg_perturbed.append(data[5][i])
                train_neg_index.append(i)
        np.save(f'{path_sentences}/X_train_neg.npy', X_train_neg_perturbed)
        np.save(f'{path_sentences}/y_train_neg.npy', y_train_neg_perturbed)
        np.save(f'{path_indexes}/train_neg_indexes.npy', train_neg_index)

        # Test positive
        X_test_pos_perturbed = []
        y_test_pos_perturbed = []
        test_pos_index = []
        for i in range(len(data[2])):
            for perturbation in perturbations:
                p_perturbed = perturbation([data[2][i]])
                X_test_pos_perturbed.append(p_perturbed[0])
                y_test_pos_perturbed.append(data[6][i])
                test_pos_index.append(i)
        np.save(f'{path_sentences}/X_test_pos.npy', X_test_pos_perturbed)
        np.save(f'{path_sentences}/y_test_pos.npy', y_test_pos_perturbed)
        np.save(f'{path_indexes}/test_pos_indexes.npy', test_pos_index)

        # Test negative
        X_test_neg_perturbed = []
        y_test_neg_perturbed = []
        test_neg_index = []
        for i in range(len(data[3])):
            for perturbation in perturbations:
                p_perturbed = perturbation([data[3][i]])
                X_test_neg_perturbed.append(p_perturbed[0])
                y_test_neg_perturbed.append(data[7][i])
                test_neg_index.append(i)
        np.save(f'{path_sentences}/X_test_neg.npy', X_test_neg_perturbed)
        np.save(f'{path_sentences}/y_test_neg.npy', y_test_neg_perturbed)
        np.save(f'{path_indexes}/test_neg_indexes.npy', test_neg_index)
    
    elif perturbation == 'vicuna':
        os.environ["REPLICATE_API_TOKEN"] = "r8_PDBpz3DCxwbqQFuLwL6vB2D9odu6HcT36mJV9"

        # Train positive
        t2 = time.time()
        tt = 0
        output_dict = {
                    'Original': [],
                    'Vicuna': [],
                    'Index': [],
                    }

        for i in range(len(data[0])):
            t1 = t2
            prompt = f'Rephrase this sentence 5 times: "{data[0][i]}".'
            output = replicate.run("replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                                    input={"prompt": prompt})
            t2 = time.time()
            td = t2 - t1
            tt = tt +  td
            if i%100 == 0:
                print(f'{i}/{len(data[0])} - {(tt/(i+1)):.1f}')
            
            output_list = []
            for item in output:
                output_list.append(item)
            output_list = ''.join(output_list)

            output_dict['Original'].append(data[0][i])
            output_dict['Vicuna'].append(output_list)
            output_dict['Index'].append(i)

            pd.DataFrame(output_dict).to_csv(f'{path_sentences}/X_train_pos_raw.csv', index=False)
        
        X_train_pos_perturbed = []
        y_train_pos_perturbed = []
        train_pos_index = []
        with open(f"src/{dataset}/perturbations/{perturbation}/sentences/X_train_pos_raw.csv") as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            for row in file:
                if row[2].isdigit():
                    i = int(row[2])
                    sentences = re.findall(r"^(?:\d+\.)\s*(.*?)[.?]", row[1], re.MULTILINE)
                    if sentences == []:
                        sentences = re.findall(r"\"(.+?)\"", row[1], re.MULTILINE)
                    for idx, sent in enumerate(sentences):
                        sentences[idx] = sent.strip('"')
                    for sentence in sentences:
                        X_train_pos_perturbed.append(sentence)
                        y_train_pos_perturbed.append(data[4][i])
                        train_pos_index.append(i)
        np.save(f'{path_sentences}/X_train_pos.npy', X_train_pos_perturbed)
        np.save(f'{path_sentences}/y_train_pos.npy', y_train_pos_perturbed)
        np.save(f'{path_indexes}/train_pos_indexes.npy', train_pos_index)

        # Train Negative
        t2 = time.time()
        tt = 0
        output_dict = {
                    'Original': [],
                    'Vicuna': [],
                    'Index': [],
                    }

        for i in range(len(data[1])):
            output_list = []
            try:
                t1 = t2
                prompt = f'Rephrase this sentence 5 times: "{data[1][i]}".'
                output = replicate.run("replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                                        input={"prompt": prompt})
                t2 = time.time()
                td = t2 - t1
                tt = tt +  td
                if i%100 == 0:
                    print(f'{i}/{len(data[1])} - {(tt/(i+1)):.1f}')

                for item in output:
                    output_list.append(item)
                output_list = ''.join(output_list)
            except:
                output_list = '404'

            output_dict['Original'].append(data[1][i])
            output_dict['Vicuna'].append(output_list)
            output_dict['Index'].append(i)

            pd.DataFrame(output_dict).to_csv(f'{path_sentences}/X_train_neg_raw.csv', index=False)
        
        X_train_neg_perturbed = []
        y_train_neg_perturbed = []
        train_neg_index = []
        with open(f"src/{dataset}/perturbations/{perturbation}/sentences/X_train_neg_raw.csv") as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            for row in file:
                if row[2].isdigit():
                    i = int(row[2])
                    sentences = re.findall(r"^(?:\d+\.)\s*(.*?)[.?]", row[1], re.MULTILINE)
                    if sentences == []:
                        sentences = re.findall(r"\"(.+?)\"", row[1], re.MULTILINE)
                    for idx, sent in enumerate(sentences):
                        sentences[idx] = sent.strip('"')
                    for sentence in sentences:
                        X_train_neg_perturbed.append(sentence)
                        y_train_neg_perturbed.append(data[5][i])
                        train_neg_index.append(i)
        np.save(f'{path_sentences}/X_train_neg.npy', X_train_neg_perturbed)
        np.save(f'{path_sentences}/y_train_neg.npy', y_train_neg_perturbed)
        np.save(f'{path_indexes}/train_neg_indexes.npy', train_neg_index)
        
        # Test positive
        t2 = time.time()
        tt = 0
        output_dict = {
                    'Original': [],
                    'Vicuna': [],
                    'Index': [],
                    }
        
        for i in range(len(data[2])):
            t1 = t2
            prompt = f'Rephrase this sentence 5 times: "{data[2][i]}".'
            output = replicate.run("replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                                    input={"prompt": prompt})
            t2 = time.time()
            td = t2 - t1
            tt = tt +  td
            if i%100 == 0:
                print(f'{i}/{len(data[2])} - {(tt/(i+1)):.1f}')
            
            output_list = []
            for item in output:
                output_list.append(item)
            output_list = ''.join(output_list)

            output_dict['Original'].append(data[2][i])
            output_dict['Vicuna'].append(output_list)
            output_dict['Index'].append(i)

            pd.DataFrame(output_dict).to_csv(f'{path_sentences}/X_test_pos_raw.csv', index=False)

        X_test_pos_perturbed = []
        y_test_pos_perturbed = []
        test_pos_index = []
        with open(f"src/{dataset}/perturbations/{perturbation}/sentences/X_test_pos_raw.csv") as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            for row in file:
                if row[2].isdigit():
                    i = int(row[2])
                    sentences = re.findall(r"^(?:\d+\.)\s*(.*?)[.?]", row[1], re.MULTILINE)
                    if sentences == []:
                        sentences = re.findall(r"\"(.+?)\"", row[1], re.MULTILINE)
                    for idx, sent in enumerate(sentences):
                        sentences[idx] = sent.strip('"')
                    for sentence in sentences:
                        X_test_pos_perturbed.append(sentence)
                        y_test_pos_perturbed.append(data[6][i])
                        test_pos_index.append(i)
        np.save(f'{path_sentences}/X_test_pos.npy', X_test_pos_perturbed)
        np.save(f'{path_sentences}/y_test_pos.npy', y_test_pos_perturbed)
        np.save(f'{path_indexes}/test_pos_indexes.npy', test_pos_index)

    # Test negative
        t2 = time.time()
        tt = 0
        output_dict = {
                    'Original': [],
                    'Vicuna': [],
                    'Index': [],
                    }
        
        for i in range(len(data[3])):
            t1 = t2
            prompt = f'Rephrase this sentence 5 times: "{data[3][i]}".'
            output = replicate.run("replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                                    input={"prompt": prompt})
            t2 = time.time()
            td = t2 - t1
            tt = tt +  td
            if i%100 == 0:
                print(f'{i}/{len(data[3])} - {(tt/(i+1)):.1f}')
            
            output_list = []
            for item in output:
                output_list.append(item)
            output_list = ''.join(output_list)

            output_dict['Original'].append(data[3][i])
            output_dict['Vicuna'].append(output_list)
            output_dict['Index'].append(i)

            pd.DataFrame(output_dict).to_csv(f'{path_sentences}/X_test_neg_raw.csv', index=False)

        X_test_neg_perturbed = []
        y_test_neg_perturbed = []
        test_neg_index = []
        with open(f"src/{dataset}/perturbations/{perturbation}/sentences/X_test_neg_raw.csv") as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            for row in file:
                if row[2].isdigit():
                    i = int(row[2])
                    sentences = re.findall(r"^(?:\d+\.)\s*(.*?)[.?]", row[1], re.MULTILINE)
                    if sentences == []:
                        sentences = re.findall(r"\"(.+?)\"", row[1], re.MULTILINE)
                    for idx, sent in enumerate(sentences):
                        sentences[idx] = sent.strip('"')
                    for sentence in sentences:
                        X_test_neg_perturbed.append(sentence)
                        y_test_neg_perturbed.append(data[7][i])
                        test_neg_index.append(i)
        np.save(f'{path_sentences}/X_test_neg.npy', X_test_neg_perturbed)
        np.save(f'{path_sentences}/y_test_neg.npy', y_test_neg_perturbed)
        np.save(f'{path_indexes}/test_neg_indexes.npy', test_neg_index)
            
    return


def embed_perturbations(dataset, perturbation, encoding_model, load_saved_perturbations):
    if load_saved_perturbations:
        X_train_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/X_train_pos.npy')
        X_train_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/X_train_neg.npy')
        X_test_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/X_test_pos.npy')
        X_test_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/X_test_neg.npy')
        y_train_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/y_train_pos.npy')
        y_train_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/y_train_neg.npy')
        y_test_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/y_test_pos.npy')
        y_test_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}/y_test_neg.npy')
    
    else:
        X_train_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/X_train_pos.npy')
        X_train_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/X_train_neg.npy')
        X_test_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/X_test_pos.npy')
        X_test_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/X_test_neg.npy')
        y_train_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/y_train_pos.npy')
        y_train_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/y_train_neg.npy')
        y_test_pos = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/y_test_pos.npy')
        y_test_neg = np.load(f'src/{dataset}/perturbations/{perturbation}/sentences/y_test_neg.npy')

        X_train_neg = np.array([i for i in X_train_neg if i])
        y_train_neg = y_train_neg[:(len(X_train_neg))]

        # Embed the sentences
        encoder = SentenceTransformer(f'{encoding_model}')
        X_train_pos = encoder.encode(X_train_pos, show_progress_bar=True)
        X_train_neg = encoder.encode(X_train_neg, show_progress_bar=True)
        X_test_pos = encoder.encode(X_test_pos, show_progress_bar=True)
        X_test_neg = encoder.encode(X_test_neg, show_progress_bar=True)

        # Rotate the data
        align_mat = load_align_mat(dataset, encoding_model, data=None, load_saved_align_mat=True)
        X_train_pos = np.matmul(X_train_pos, align_mat)
        X_train_neg = np.matmul(X_train_neg, align_mat)
        X_test_pos = np.matmul(X_test_pos, align_mat)
        X_test_neg = np.matmul(X_test_neg, align_mat)

        # Save the rotated embedded sentences and labels
        path = f'src/{dataset}/perturbations/{perturbation}/embeddings/{encoding_model}'
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


# Character perturbations
def return_random_number(begin, end):
    return randint(begin, end)


def return_adjacent_char(input_char):
    
    if (input_char == 'a'):
        return 's'
    
    elif (input_char == 'b'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'v'
        else:
            return 'n'
        
    elif (input_char == 'c'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'x'
        else:
            return 'v'
        
    elif (input_char == 'd'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 's'
        else:
            return 'f'
        
    elif (input_char == 'e'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'w'
        else:
            return 'r'
        
    elif (input_char == 'f'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'd'
        else:
            return 'g'
        
    elif (input_char == 'g'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'f'
        else:
            return 'h'
        
    elif (input_char == 'h'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'g'
        else:
            return 'j'
        
    elif (input_char == 'i'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'u'
        else:
            return 'o'
        
    elif (input_char == 'j'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'h'
        else:
            return 'k'
        
    elif (input_char == 'k'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'j'
        else:
            return 'l'
    
    elif (input_char == 'l'):
        return 'k'
        
    elif (input_char == 'm'):
        return 'n'
        
    elif (input_char == 'n'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'b'
        else:
            return 'm'
        
    elif (input_char == 'o'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'i'
        else:
            return 'p'
        
    elif (input_char == 'p'):
        return 'o'
    
    elif (input_char == 'q'):
        return 'w'
        
    elif (input_char == 'r'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'e'
        else:
            return 't'
        
    elif (input_char == 's'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'a'
        else:
            return 'd'
        
    elif (input_char == 't'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'r'
        else:
            return 'y'
        
    elif (input_char == 'u'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'y'
        else:
            return 'i'
    
    elif (input_char == 'v'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'c'
        else:
            return 'b'
        
    elif (input_char == 'w'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'q'
        else:
            return 'e'
        
    elif (input_char == 'x'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'z'
        else:
            return 'c'
        
    elif (input_char == 'y'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 't'
        else:
            return 'u'
        
    elif (input_char == 'z'):
        return 'x'
    #---------------------------------------------
    elif (input_char == 'A'):
        return 'S'
    
    elif (input_char == 'B'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'V'
        else:
            return 'N'
        
    elif (input_char == 'C'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'X'
        else:
            return 'V'
        
    elif (input_char == 'D'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'S'
        else:
            return 'F'
        
    elif (input_char == 'E'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'W'
        else:
            return 'R'
        
    elif (input_char == 'F'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'D'
        else:
            return 'G'
        
    elif (input_char == 'G'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'F'
        else:
            return 'H'
        
    elif (input_char == 'H'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'G'
        else:
            return 'J'
        
    elif (input_char == 'I'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'U'
        else:
            return 'O'
        
    elif (input_char == 'J'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'H'
        else:
            return 'K'
        
    elif (input_char == 'K'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'J'
        else:
            return 'L'
    
    elif (input_char == 'L'):
        return 'K'
        
    elif (input_char == 'M'):
        return 'N'
        
    elif (input_char == 'N'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'B'
        else:
            return 'M'
        
    elif (input_char == 'O'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'I'
        else:
            return 'P'
        
    elif (input_char == 'P'):
        return 'O'
    
    elif (input_char == 'Q'):
        return 'W'
        
    elif (input_char == 'R'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'E'
        else:
            return 'T'
        
    elif (input_char == 'S'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'A'
        else:
            return 'D'
        
    elif (input_char == 'T'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'R'
        else:
            return 'Y'
        
    elif (input_char == 'U'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Y'
        else:
            return 'I'
    
    elif (input_char == 'V'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'C'
        else:
            return 'B'
        
    elif (input_char == 'W'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Q'
        else:
            return 'E'
        
    elif (input_char == 'X'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Z'
        else:
            return 'C'
        
    elif (input_char == 'Y'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'T'
        else:
            return 'U'
        
    elif (input_char == 'Z'):
        return 'X'
    
    else:
        return '*'


def swap_characters(input_word, position, adjacent):
    temp_word = ''
    if (adjacent == 'left'):
        if (position == 1):
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif (position == len(input_word)-1):
            temp_word = input_word[0:position-1]
            temp_word += input_word[position]
            temp_word += input_word[position-1]
        elif (position > 1 and position < len(input_word)-1):
            temp_word = input_word[0:position-1]
            temp_word += input_word[position]
            temp_word += input_word[position-1]
            temp_word += input_word[position+1:]
            
    elif (adjacent == 'right'):
        if (position == 0):
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif (position == len(input_word)-2):
            temp_word = input_word[0:position]
            temp_word += input_word[position+1]
            temp_word += input_word[position]
        elif (position > 0 and position < len(input_word)-2):
            temp_word = input_word[0:position]
            temp_word += input_word[position+1]
            temp_word += input_word[position]
            temp_word += input_word[position+2:]
            
    return temp_word


def char_replacement(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if has_3_letter_word == False:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True

        #print('Sentence: ', sample_text)
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        char_is_letter = False
        tries_number = 0
        
        while (char_is_letter != True and tries_number <= 20):
            random_char_index = return_random_number(1, len(selected_word)-2)
            tries_number += 1
            if ((ord(selected_word[random_char_index]) >= 97 and ord(selected_word[random_char_index]) <= 122) or (ord(selected_word[random_char_index]) >= 65 and ord(selected_word[random_char_index]) <= 90)):
                char_is_letter = True
                is_sample_perturbed = True
        
        #print('Random position:', random_char_index)
        #print('Character to replace:', selected_word[random_char_index])
        
        #--------------------------- replace the character
    
        char_to_replace = selected_word[random_char_index]
        adjacent_char = return_adjacent_char(char_to_replace)
        #print('Adjacent character:', adjacent_char)
        
        temp_word = selected_word[:random_char_index]
        temp_word += adjacent_char
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After replacement:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)
    return np.array(new_sentences)
    

def char_swapping(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True

        #print('Sentence: ', sample_text)
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        random_char_index = return_random_number(0, len(selected_word)-1)

        #print('Random position:', random_char_index)
        #print('Char in random position:', selected_word[random_char_index])
        
        #--------------------------- select an adjacent for swapping
            
        adjacent_for_swapping = ''
        
        if (random_char_index == 0):
            adjacent_for_swapping = 'right'
        elif (random_char_index == len(selected_word)-1):
            adjacent_for_swapping = 'left'
        else:
            adjacent = return_random_number(1, 2)
            if(adjacent == 1):
                adjacent_for_swapping = 'left'
            else:
                adjacent_for_swapping = 'right'
                
        #print('Adjacent for swapping:', adjacent_for_swapping)

        #--------------------------- swap the character and the adjacent
            
        temp_word = swap_characters(selected_word, random_char_index, adjacent_for_swapping)
        perturbed_word = ""

        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After swapping:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def char_deletion(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = return_random_number(1, len(selected_word)-2)
        #print('Random position:', random_char_index)
        #print('Character to delete:', selected_word[random_char_index])
        
        #--------------------------- delete the character
    
        temp_word = selected_word[:random_char_index]
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After deletion:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def char_insertion(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = return_random_number(1, len(selected_word)-2)
        #print('Random position:', random_char_index)
        
        #--------------------------- select a random character
        
        random_char_code = return_random_number(97, 122)
        #print('Random character:', chr(random_char_code))
    
        temp_word = selected_word[:random_char_index]
        temp_word += chr(random_char_code)
        temp_word += selected_word[random_char_index:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After insertion:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def char_repetition(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 2):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        #--------------------------- select a random position
        
        selected_word = sample_tokenized[random_word_index]
        
        random_char_index = return_random_number(1, len(selected_word)-2)
        #print('Random position:', random_char_index)
        #print('Character to repeat:', selected_word[random_char_index])
        
        #--------------------------- repeat the character
    
        temp_word = selected_word[:random_char_index]
        temp_word += selected_word[random_char_index] + selected_word[random_char_index]
        temp_word += selected_word[random_char_index+1:]
        
        perturbed_word = ""
        for i in range(0, len(temp_word)):
            perturbed_word += temp_word[i]
        
        #print('After repetition:', perturbed_word)
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += perturbed_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


# Word perturbations
def change_ordering(input_length, input_side, input_changes):
    ordering = []
    
    if (input_side == 1):
        for i in range(0, input_length):
            if (i < input_changes):
                
                candidates=[]
                for j in range(0, input_changes):
                    if (j != i and j not in ordering):
                        candidates.append(j)
                        
                if (len(candidates) > 0):
                    random_index = return_random_number(0, len(candidates)-1)
                    ordering.append(candidates[random_index])
                else:
                    ordering.append(i)
            else:
                ordering.append(i)
                
    elif (input_side == 2):
        for i in range(0, input_length):
            if (i < input_length-input_changes):
                ordering.append(i)
                
            else:
                candidates=[]
                for j in range(input_length-input_changes, input_length):
                    if (j != i and j not in ordering):
                        candidates.append(j)
                        
                if (len(candidates) > 0):
                    random_index = return_random_number(0, len(candidates)-1)
                    ordering.append(candidates[random_index])
                else:
                    ordering.append(i)
                        
    return ordering


def is_third_person(input_pos_tag):
    subject = ''
    for i in range(0, len(input_pos_tag)):
        token = input_pos_tag[i]
        if (subject == ''):
            if (token[0].lower() in ('it', 'this', 'that', 'he', 'she')):
                subject = 'third person'
            elif (token[1] in ('NNP')):
                subject = 'third person'
            elif (token[0].lower() in ('i', 'we', 'you', 'they', 'she', 'these', 'those')):
                subject = 'not third person'
            elif (token[0].lower() in ('NNPS')):
                subject = 'not third person'
    if (subject == 'third person'):
        return 'third person'
    elif (subject == 'not third person'):
        return 'not third person'
    else:
        return 'none'


def word_deletion(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue
    
        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 1):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
            
        is_sample_perturbed = True
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)
    return np.array(new_sentences)
    

def word_negation(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        sample_pos_tag = nltk.pos_tag(sample_tokenized)
            
        #print(sample_pos_tag)
        
        Perturbed_sample = ""
        
        remove_negation = False
        basic_to_third_person = False
        can_change_basic_form = True
        can_change_pp_modal = True
        can_change_future = True
        basic_to_past = False
        
        for i in range(0, len(sample_pos_tag)):
            token = sample_pos_tag[i]
            #print(token[0], token[1])
            if (remove_negation == False and basic_to_third_person == False and can_change_basic_form == True and basic_to_past == False and can_change_pp_modal == True and i < len(sample_pos_tag) - 1):
                if (token[0] == 'has' and sample_pos_tag[i+1][1] == 'VBN'): #----- third person singular present perfect
                    verb = token[0]
                    verb = verb + ' not'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[0] == 'have' and sample_pos_tag[i+1][1] == 'VBN'): #----- present perfect
                    verb = token[0]
                    verb = verb + ' not'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[0] == 'had' and sample_pos_tag[i+1][1] == 'VBN'): #----- past perfect
                    verb = token[0]
                    verb = verb + ' not'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                
                elif (token[0] == 'does' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, third person present simple
                    remove_negation = True
                    basic_to_third_person = True
                    Perturbed_sample += ""
                    is_sample_perturbed = True
                    
                elif (token[0] == 'do' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present simple
                    remove_negation = True
                    can_change_basic_form = False
                    Perturbed_sample += ""
                    is_sample_perturbed = True
                    
                elif (token[0] == 'did' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, past simple
                    remove_negation = True
                    can_change_basic_form = True
                    basic_to_past = True
                    Perturbed_sample += ""
                    is_sample_perturbed = True
                    
                elif (token[0] == 'has' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, third person present perfect
                    remove_negation = True
                    can_change_pp_modal = False
                    Perturbed_sample += "has" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'have' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present perfect
                    remove_negation = True
                    can_change_pp_modal = False
                    Perturbed_sample += "have" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'had' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present perfect
                    remove_negation = True
                    can_change_pp_modal = False
                    Perturbed_sample += "had" + ' '
                    is_sample_perturbed = True
                    
                elif (token[1] == 'MD' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, future and modal verbs
                    verb = token[0]
                    if (verb == 'ca'):
                        verb = 'can'
                    remove_negation = True
                    can_change_pp_modal = False
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] in ('is', 'are', 'was', 'were', 'am') and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    verb = token[0]
                    remove_negation = True
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                    
                elif (token[1] == 'MD'): #----- future and modal verbs
                    verb = token[0]
                    if (verb == 'can' or verb == 'Can'):
                        verb = verb + 'not'
                    else:
                        verb = verb + ' not'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] in ('is', 'are', 'was', 'were', 'am')): #----- to be present and past, continuous present and past
                    verb = token[0]
                    verb = verb + ' not'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
            
                elif (token[1] == 'VBZ'): #----- third person singular present
                    verb = token[0]
                    length = len(verb)
                    if (verb == 'has'):
                        verb = 'does not ' + 'have'
                    elif (verb[length-3:] == 'oes'):
                        verb = 'does not ' + verb[:length-2]
                    elif (verb[length-4:] == 'ches'):
                        verb = 'does not ' + verb[:length-2]
                    elif (verb[length-3:] == 'ses'):
                        verb = 'does not ' + verb[:length-2]
                    elif (verb[length-4:] == 'shes'):
                        verb = 'does not ' + verb[:length-2]
                    elif (verb[length-3:] == 'xes'):
                        verb = 'does not ' + verb[:length-2]
                    elif (verb[length-3:] == 'zes'):
                        verb = 'does not ' + verb[:length-2]
                    elif (verb[length-3:] == 'ies'):
                        verb = 'does not ' + verb[:length-3] + 'y'
                    else:
                        verb = 'does not ' + verb[:length-1]
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBP'): #----- basic form present
                    verb = token[0]
                    length = len(verb)
                    verb = 'do not ' + verb
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBD'): #----- past
                    verb = token[0]
                    verb = 'did not ' + WordNetLemmatizer().lemmatize(verb,'v')
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                else:
                    Perturbed_sample += token[0] + ' '
                    
            elif (remove_negation == True):
                if (token[0] in ('not', "n't")): #----- removing not after do or does
                    Perturbed_sample += ""
                    remove_negation = False
                    
            elif (basic_to_third_person == True and can_change_basic_form == True):
                if (token[1] == 'VB'): #----- converting basic form to third person singular
                    verb = token[0]
                    length = len(verb)
                    if (verb == 'have'):
                        verb = 'has'
                    elif (verb == 'go'):
                        verb = 'goes'
                    elif (verb[length-2:] == 'ch'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 's'):
                        verb = verb + 'es'
                    elif (verb[length-2:] == 'sh'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 'x'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 'z'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 'y'):
                        verb = verb[:length-1] + 'ies'
                    else:
                        verb = verb + 's'
                    
                    Perturbed_sample += verb + ' '
                    basic_to_third_person = False
                    
            elif (can_change_basic_form == False):
                if (token[1] == 'VB'): #----- do not change basic form
                    verb = token[0]
                    Perturbed_sample += verb + ' '
                    can_change_basic_form = True
                    
            elif (can_change_basic_form == True and basic_to_past == True):
                if (token[1] == 'VB'): #----- change basic form to past
                    verb = token[0]
                    past_tense = ""
                    
                    default_conjugator = mlconjug3.Conjugator(language='en')
                    past_verb = default_conjugator.conjugate(verb)
                    all_conjugates = past_verb.iterate()
                    
                    for j in range(0, len(all_conjugates)):
                        if (all_conjugates[j][1] == 'indicative past tense'):
                            past_tense = all_conjugates[j][3]
                    
                    Perturbed_sample += past_tense + ' '
                    basic_to_past = False
                    
            elif (can_change_pp_modal == False):
                if (token[1] in ('VBN', 'VB')): #----- keep past participle or modal or will
                    verb = token[0]
                    Perturbed_sample += verb + ' '
                    can_change_pp_modal = True

            else:
                verb = token[0]
                Perturbed_sample += verb + ' '
                    
        
        
        #print('Perturbed sample:', Perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(Perturbed_sample)

    return np.array(new_sentences)


def word_ordering(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)

        perturbed_sample = ""
            
        if (len(sample_tokenized) > 3):
            #print('Sample can be perturbed.')
            
            last_token = ""
            if (sample_tokenized[len(sample_tokenized)-1] in ('.', '?', '!', ';', ',')):
                last_token = sample_tokenized[len(sample_tokenized)-1]
                sample_tokenized = sample_tokenized[0: len(sample_tokenized)-1]
            
            ordering_side = return_random_number(1, 2)
            
            # if (ordering_side == 1): #----- change word ordering in the beginning
            #     print('Change ordering side: Beginning')
            # elif (ordering_side == 2): #----- change word ordering in the end
            #     print('Change ordering side: End')
                
            num_changed_words = return_random_number(2, len(sample_tokenized)-1)
            #print('Number of words for changing the order:', num_changed_words)
                
            new_word_order = change_ordering(len(sample_tokenized), ordering_side, num_changed_words)
                
            #print('New word order:', new_word_order)
            
            for i in range(0, len(new_word_order)):
                temp_index = new_word_order[i]
                perturbed_sample += sample_tokenized[temp_index] + ' '
            perturbed_sample += last_token
            
            is_sample_perturbed = True
            
        else:
            perturbed_sample = sample_text
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def word_repetition(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        random_word_index = 0
        random_word_selected = False

        # Check if the sentence has at least a 3 letter word, otherwise skip
        has_3_letter_word = False
        for word in sample_tokenized:
            if len(word) >= 3:
                has_3_letter_word = True
        if not has_3_letter_word:
            new_sentences.append(sample_text)
            continue

        while (random_word_selected != True):
            random_word_index = return_random_number(0, len(sample_tokenized)-1)
            if (len(sample_tokenized[random_word_index]) > 1):
                random_word_selected = True
    
        #print('Selected random word:', sample_tokenized[random_word_index])
        
        
        
        selected_word = sample_tokenized[random_word_index]
        
        
        #--------------------------- reconstruct the perturbed sample
        
        perturbed_sample = ""
        
        for i in range(0, random_word_index):
                
            perturbed_sample += sample_tokenized[i] + ' '
            
        perturbed_sample += selected_word + ' ' + selected_word + ' '
        is_sample_perturbed = True
        
        for i in range(random_word_index+1, len(sample_tokenized)):    
            perturbed_sample += sample_tokenized[i] + ' '
        
        #print('Perturbed sample:', perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(perturbed_sample)

    return np.array(new_sentences)


def word_singular_plural_verb(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        sample_pos_tag = nltk.pos_tag(sample_tokenized)
        
        #print(sample_pos_tag)
        
        Perturbed_sample = ""
        
        remove_negation = False
        
        for i in range(0, len(sample_pos_tag)):
            token = sample_pos_tag[i]
            #print(token[0], token[1])
            if (remove_negation == False) and i < len(sample_pos_tag) - 1:
                if (token[0] == 'has' and sample_pos_tag[i+1][1] == 'VBN'): #----- third person singular present perfect
                    verb = 'have'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[0] == 'have' and sample_pos_tag[i+1][1] == 'VBN'): #----- present perfect
                    verb = 'has'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[0] == 'does' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, third person present simple
                    verb = 'do not'
                    remove_negation = True
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'do' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present simple
                    verb = 'does not'
                    remove_negation = True
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'has' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, third person present perfect
                    remove_negation = True
                    Perturbed_sample += "have not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'have' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present perfect
                    remove_negation = True
                    Perturbed_sample += "has not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'is' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += 'are not' + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'are' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += 'is not' + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'was' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += 'were not' + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'were' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += 'was not' + ' '
                    is_sample_perturbed = True
                
                elif (token[0] == 'does'): #----- negative, third person present simple
                    verb = 'do'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'do'): #----- negative, present simple
                    verb = 'does'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[0] == 'is'): #----- to be present and past, continuous present and past
                    Perturbed_sample += 'are' + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'are'): #----- to be present and past, continuous present and past
                    Perturbed_sample += 'is' + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'was'): #----- to be present and past, continuous present and past
                    Perturbed_sample += 'were' + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'were'): #----- to be present and past, continuous present and past
                    Perturbed_sample += 'was' + ' '
                    is_sample_perturbed = True
            
                elif (token[1] == 'VBZ'): #----- third person singular present
                    verb = token[0]
                    length = len(verb)
                    if (verb == 'has'):
                        verb = 'have'
                    elif (verb[length-3:] == 'oes'):
                        verb = verb[:length-2]
                    elif (verb[length-4:] == 'ches'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'ses'):
                        verb = verb[:length-2]
                    elif (verb[length-4:] == 'shes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'xes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'zes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'ies'):
                        verb = verb[:length-3] + 'y'
                    else:
                        verb = verb[:length-1]
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBP'): #----- basic form present
                    verb = token[0]
                    length = len(verb)
                    if (verb == 'have'):
                        verb = 'has'
                    elif (verb == 'go'):
                        verb = 'goes'
                    elif (verb[length-2:] == 'ch'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 's'):
                        verb = verb + 'es'
                    elif (verb[length-2:] == 'sh'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 'x'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 'z'):
                        verb = verb + 'es'
                    elif (verb[length-1:] == 'y'):
                        verb = verb[:length-1] + 'ies'
                    else:
                        verb = verb + 's'
                    Perturbed_sample += verb + ' '
                    is_sample_perturbed = True
                
                else:
                    Perturbed_sample += token[0] + ' '
                    
            elif (remove_negation == True):
                if (token[0] in ('not', "n't")): #----- removing not after do or does
                    Perturbed_sample += ""
                    remove_negation = False
                    
            else:
                verb = token[0]
                Perturbed_sample += verb + ' '
        
        #print('Perturbed sample:', Perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(Perturbed_sample)

    return np.array(new_sentences)


def word_verb_tense(input_data):
    new_sentences = []
    for sample_text in input_data:
        is_sample_perturbed = False
        sample_tokenized = nltk.word_tokenize(sample_text)
        sample_pos_tag = nltk.pos_tag(sample_tokenized)
        
        #print(sample_pos_tag)
        
        Perturbed_sample = ""
        
        remove_negation = False
        can_change_basic_form = True
        
        for i in range(0, len(sample_pos_tag)):
            token = sample_pos_tag[i]
            #print(token[0], token[1])
            if (remove_negation == False and can_change_basic_form == True and i < len(sample_pos_tag) - 1):
                
                if (token[0] == 'does' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, third person present simple
                    remove_negation = True
                    Perturbed_sample += "did not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'do' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, present simple
                    remove_negation = True
                    Perturbed_sample += "did not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'did' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, past simple
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        remove_negation = True
                        can_change_basic_form = False
                        Perturbed_sample += "does not" + ' '
                        is_sample_perturbed = True
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        remove_negation = True
                        Perturbed_sample += "do not" + ' '
                        is_sample_perturbed = True
                    
                elif (token[0] in ('is', 'am') and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += "was not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'are' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += "were not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'was' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        remove_negation = True
                        Perturbed_sample += "is not" + ' '
                        is_sample_perturbed = True
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        remove_negation = True
                        Perturbed_sample += "am not" + ' '
                        is_sample_perturbed = True
                        
                elif (token[0] == 'were' and sample_pos_tag[i+1][0] in ('not', "n't")): #----- negative, to be present and past, continuous present and past
                    remove_negation = True
                    Perturbed_sample += "are not" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] in ('is', 'am')): #----- to be present and past, continuous present and past
                    Perturbed_sample += "was" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'are'): #----- to be present and past, continuous present and past
                    Perturbed_sample += "were" + ' '
                    is_sample_perturbed = True
                    
                elif (token[0] == 'was'): #----- to be present and past, continuous present and past
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        Perturbed_sample += "is" + ' '
                        is_sample_perturbed = True
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        Perturbed_sample += "am" + ' '
                        is_sample_perturbed = True
                        
                elif (token[0] == 'were'): #----- to be present and past, continuous present and past
                    Perturbed_sample += "are" + ' '
                    is_sample_perturbed = True
            
                elif (token[1] == 'VBZ'): #----- third person singular present
                    verb = token[0]
                    length = len(verb)
                    if (verb == 'has'):
                        verb = 'have'
                    elif (verb[length-3:] == 'oes'):
                        verb = verb[:length-2]
                    elif (verb[length-4:] == 'ches'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'ses'):
                        verb = verb[:length-2]
                    elif (verb[length-4:] == 'shes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'xes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'zes'):
                        verb = verb[:length-2]
                    elif (verb[length-3:] == 'ies'):
                        verb = verb[:length-3] + 'y'
                    else:
                        verb = verb[:length-1]
                        
                    past_tense = ""
                    
                    default_conjugator = mlconjug3.Conjugator(language='en')
                    past_verb = default_conjugator.conjugate(verb)
                    all_conjugates = past_verb.iterate()
                    
                    for j in range(0, len(all_conjugates)):
                        if (all_conjugates[j][1] == 'indicative past tense'):
                            past_tense = all_conjugates[j][3]
                    
                    Perturbed_sample += past_tense + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBP'): #----- basic form present
                    verb = token[0]
                    
                    past_tense = ""
                    
                    default_conjugator = mlconjug3.Conjugator(language='en')
                    past_verb = default_conjugator.conjugate(verb)
                    all_conjugates = past_verb.iterate()
                    
                    for j in range(0, len(all_conjugates)):
                        if (all_conjugates[j][1] == 'indicative past tense'):
                            past_tense = all_conjugates[j][3]
                    
                    Perturbed_sample += past_tense + ' '
                    is_sample_perturbed = True
                
                elif (token[1] == 'VBD'): #----- past
                    if (is_third_person(sample_pos_tag) == 'third person'):
                        verb = token[0]
                        verb = WordNetLemmatizer().lemmatize(verb,'v')
                        
                        length = len(verb)
                        if (verb == 'have'):
                            verb = 'has'
                        elif (verb == 'go'):
                            verb = 'goes'
                        elif (verb == 'do'):
                            verb = 'does'
                        elif (verb[length-2:] == 'ch'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 's'):
                            verb = verb + 'es'
                        elif (verb[length-2:] == 'sh'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 'x'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 'z'):
                            verb = verb + 'es'
                        elif (verb[length-1:] == 'y'):
                            verb = verb[:length-1] + 'ies'
                        else:
                            verb = verb + 's'
                        
                        Perturbed_sample += verb + ' '
                        is_sample_perturbed = True
                        
                    elif (is_third_person(sample_pos_tag) == 'not third person'):
                        verb = token[0]
                        verb = WordNetLemmatizer().lemmatize(verb,'v')
                        
                        Perturbed_sample += verb + ' '
                        is_sample_perturbed = True
                
                else:
                    Perturbed_sample += token[0] + ' '
                    
            elif (remove_negation == True):
                if (token[0] in ('not', "n't")): #----- removing not after do or does
                    Perturbed_sample += ""
                    remove_negation = False
                    
            elif (can_change_basic_form == False):
                if (token[1] == 'VB'): #----- do not change basic form
                    verb = token[0]
                    Perturbed_sample += verb + ' '
                    can_change_basic_form = True
                    
            else:
                verb = token[0]
                Perturbed_sample += verb + ' '
        
        #print('Perturbed sample:', Perturbed_sample)
        #print('----------------------------------------------------------')
        new_sentences.append(Perturbed_sample)
    return np.array(new_sentences)
