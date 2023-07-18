ANTONIO - Abstract iNterpreTation fOr Nlp verIficatiOn
========

Structure
------------
```
.
├── results                                          - folder containing all the final results
├── src
│   ├── medical
│   │   ├── data                                     - folder containing the dataset
│   │   ├── embeddings                               - folder containing the embedded data
│   │   ├── hyperrectangles                          - folder containing the hyper-rectangles
│   │   ├── models
│   │   │   ├── onnx                                 - folder containing the onnx networks
│   │   │   └── tf                                   - folder containing the tensorflow networks
│   │   └── perturbations                            - folder containing the perturbations and their embeddings
│   │
│   ├── ruarobot
│   │   ├── data                                     - folder with the dataset
│   │   ├── embeddings                               - folder with the embedded data
│   │   ├── hyperrectangles                          - folder with the hyper-rectangles
│   │   ├── models
│   │   │   ├── onnx                                 - folder with the onnx networks
│   │   │   └── tf                                   - folder with the tensorflow networks
│   │   └── perturbations                            - folder containing the perturbations and their embeddings
│   │
│   ├── data.py                                      - file for loading and processing the data
│   ├── hyperrectangles.py                           - file for creating hyper-rectangles
│   ├── perturbations.py                             - file for creating, saving and embedding the perturbations
│   ├── results.py                                   - file for calculating results
│   └── train.py                                     - file for training the networks
│
├── verification
│   ├── marabou
│   │   ├── medical                                  - folder containing marabou properties for this dataset
│   │   ├── ruarobot                                 - folder containing marabou properties for this dataset
│   │   ├── outputs                                  - folder containing marabou outputs
│   │   ├── marabout_property_parser.py              - file for creating the marabou properties
│   │   └── marabout_queries.py                      - file for creating the marabou queries
│
├── example.py                                       - file for reproducing the paper's experiments
├── requirements.txt                                 - pip requirements
└── tf2onnx.sh                                       - script for creating onnx networks from tensorflow
```

System Requirements
------------
Ubuntu 18.04 (64-bit), Python 3.6 or higher.

Installation
------------
Install the dependencies:
```
pip3 install -r requirements.txt
```
To reproduce the example also install:
* [Marabou](https://github.com/eth-sri/eran)
* [Replicate](https://replicate.com/) (This API requires a subscription)

Reproducing Experiments and Results
-------------
To process the data, create the hyper-rectangles, train the networks, create the Marabou queries and calculate the metrics, run:
```
python3 main.py
```
