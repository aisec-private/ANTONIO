import Marabou
import MarabouNetworkONNX
import os


# Set the Marabou option to restrict printing
options = Marabou.createOptions(verbosity = 0)
datasets = ['ruarobot', 'medical']

for dataset in datasets:
    network_directory = f'../../src/{dataset}/models/onnx'
    properties_directory = f'./{dataset}/properties'

    # Set input bounds
    for network_filename in os.listdir(network_directory):
        f = os.path.join(network_directory, network_filename)
        short_name = network_filename.split('.')[0]

        # checking if it is a file
        if os.path.isfile(f) and len(short_name):
            for directory in os.listdir(properties_directory):
                folder = os.path.join(properties_directory, directory)
                new_directory_name = short_name+"-"+folder.split('/')[-1]
                os.mkdir(os.path.join(f'./{dataset}/queries', new_directory_name))

                for property_filename in os.listdir(folder):
                    # checking if it is a file
                    if os.path.isfile(os.path.join(folder, property_filename)):
                        network = Marabou.read_onnx(f)

                        # Get the input and output variable numbers; [0] since first dimension is batch size
                        inputVars = network.inputVars[0][0]
                        outputVars = network.outputVars[0][0]
                        with open(os.path.join(folder, property_filename)) as f2:
                            for line in f2.readlines():
                                split = line.split()
                                if line.__contains__("x") and line.__contains__("<="):
                                    network.setUpperBound(inputVars[int(split[0][1:])], float(split[2]))
                                elif line.__contains__("x") and line.__contains__(">="):
                                    network.setLowerBound(inputVars[int(split[0][1:])], float(split[2]))

                            # y_correct - y_i <= 0
                            network.addInequality([outputVars[0], outputVars[1]], [1, -1], 0, isProperty=True)
                            network.saveQuery(filename=(f'./{dataset}/queries'+new_directory_name+"/"+property_filename.split('@')[-1]+".ipq"))


