from src.hyperrectangles import load_hyperrectangles
import numpy as np
import shutil
import os


def parse_properties(datasets, encoding_models, h_names):
    for dataset in datasets:
        for encoding_model, encoding_name in encoding_models.items():
            for h, h_name in h_names.items():
                hyperrectangles = load_hyperrectangles(dataset, encoding_model, h_name, load_saved_hyperrectangles=True)

                if len(h_name) > 1:
                    h_name = ['perturbations']
                h_name = h_name[0]
                print(f'{dataset} -|- {encoding_name} -|-  -|- {hyperrectangles.shape} -|- {h_name}')

                models_directory = f'./verification/marabou/{dataset}/{encoding_name}/models'
                if not os.path.exists(models_directory):
                    os.makedirs(models_directory)
                
                model_path = f'./src/{dataset}/models/onnx/{encoding_name}'
                for model in os.listdir(model_path):
                    shutil.copy(os.path.join(model_path, model), models_directory)
                
                properties_directory = f'./verification/marabou/{dataset}/{encoding_name}/properties/{h_name}'
                if not os.path.exists(properties_directory):
                    os.makedirs(properties_directory)

                for i, cube in enumerate(hyperrectangles):
                    with open(f'{properties_directory}/{h_name}@{i}', 'w') as property_file:
                        for j,d in enumerate(cube):
                            property_file.write("x" + str(j) + " >= " + str(d[0]) + "\n")
                            property_file.write("x" + str(j) + " <= " + str(d[1]) + "\n")
                        property_file.write("y0 <= y1")
