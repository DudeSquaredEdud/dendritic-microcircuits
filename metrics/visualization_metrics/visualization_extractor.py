"""
# Visualization Extractor
By Ashton Andrepont
"""

import numpy as np
from .neuronparse import NeuronParser


class VisualExtractor:
    """
    # Visual Extractor
    Extracts the data needed for Theresa's model visualization.
    """

    def __init__(self, file_name: str):
        self.pyramidal_weights = np.array([])
        self.interneuron_weights = np.array([])
        self.output_file_p = open((file_name + "_pyramidal.txt"), "a")
        self.output_file_i = open((file_name + "_interneuron.txt"), "a")
        np.set_printoptions(suppress=True, threshold=np.inf)
        self.neuron_parser = NeuronParser()

    def layers_to_file(self, layers: list) -> None:
        """
        Writes a list of layers to a file.

        This is a raw text file that is layer handled by the neuron_parse.py


        """
        # First for the pyramidal neurons
        for l in layers:
            self.__pyramidal_data_extraction(l)
            self.output_file_p.write(np.array2string(self.pyramidal_weights) + "\n")
            self.pyramidal_weights = np.array([])

        for l in layers:
            self.__interneuron_data_extraction(l)
            self.output_file_i.write(np.array2string(self.interneuron_weights) + "\n")
            self.interneuron_weights = np.array([])

    def close(self) -> None:
        """
        Converts the files into JSONs and loses the extractor.

        Please remember to run this at the end.
        """
        for file, mode in [[self.output_file_i, "i"], [self.output_file_p, "p"]]:
            self.neuron_parser.file_to_delta(file.name, mode)
            file.close()

    def __pyramidal_data_extraction(self, layer) -> None:
        self.pyramidal_weights = np.append(self.pyramidal_weights, layer.id_num)
        for neuronNum, neuron in enumerate(layer.pyrs):
            self.pyramidal_weights = np.append(self.pyramidal_weights, neuron.id_num)
            self.pyramidal_weights = np.append(self.pyramidal_weights, neuron.apical_mp)
            self.pyramidal_weights = np.append(self.pyramidal_weights, neuron.basal_mp)
            self.pyramidal_weights = np.append(self.pyramidal_weights, neuron.soma_mp)
            self.pyramidal_weights = np.append(self.pyramidal_weights, neuron.soma_act)

    def __interneuron_data_extraction(self, layer) -> None:
        self.interneuron_weights = np.append(self.interneuron_weights, layer.id_num)
        for neuronNum, neuron in enumerate(layer.inhibs):
            self.interneuron_weights = np.append(
                self.interneuron_weights, neuron.id_num
            )
            self.interneuron_weights = np.append(
                self.interneuron_weights, neuron.soma_mp
            )
            self.interneuron_weights = np.append(
                self.interneuron_weights, neuron.soma_act
            )
            self.interneuron_weights = np.append(
                self.interneuron_weights, neuron.dend_mp
            )
