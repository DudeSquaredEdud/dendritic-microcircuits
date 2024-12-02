"""
# Neuron Parse
By Ashton Andrepont

I would have named it neuron_parse but *for some reason* it's a reserved name so *whatever I guess*.

Takes in a text file of raw numpy arrays of training data from the 
[Dendritic Microcircuits](https://github.com/AnthonySMaida/dendritic-microcircuits) 
implementation created under professor Anthony S. Maida.

### file_to_delta(file_name, mode):
Takes in a file name and a mode ("p" for pyramidal, "i" for interneuron).
Converts it to a `json` of training data.
"""

import re
import json
from typing import List
import numpy as np


class NeuronParser:

    def __file_to_list_of_arrays(self, file_name: str) -> List[np.ndarray]:
        """Convert a file into a list of np.arrays."""

        def get_complete_line(file) -> str:
            """Reads lines from a file until a complete array string is formed."""
            # Get a line (Remove the \n, because readLine picks it up.)
            line: str = file.readline().removesuffix("\n")
            if not line:  # unless we're at the end
                return ""
            # if it's not the end of an array...
            while not line.endswith("]"):
                # get the next line.
                next_line: str = file.readline().removesuffix("\n")
                if not next_line:  # unless we're at the end.
                    break
                # Add a space! Then add the next line.
                # This space exorcises parsing demons.
                line += " " + next_line
            return line.strip()

        def parse_into_np_array(line: str) -> np.array:
            """Turns a line that looks like an array into an actual np.array."""
            output_array = np.array([])

            # Pretty array
            numbers = np.array(
                re.sub(r"\.\,", ",", re.sub(r"\n", "", re.sub(r" +", r", ", line)))
                .replace("[", "")
                .replace("]", "")
                .split(", ")
            )

            # Parsing demon exorcism
            for number in numbers:
                if number and not number.endswith("1."):
                    output_array = np.append(output_array, float(number))
                elif number:
                    output_array = np.append(
                        output_array, float(number.removesuffix("1."))
                    )

            return output_array

        # The Arrays are parsed into a set of np.arrays.
        parsed_arrays = []
        with open(file_name, "r") as file:
            while True:
                # Get a line...
                line = get_complete_line(file)
                # ...unless you don't.
                if not line:
                    break
                # Parse it and put it away.
                parsed_arrays.append(parse_into_np_array(line))

        # Then they're parsed into their layers
        # Get the number of layers using a classic max
        layer_amount = 0
        for array in parsed_arrays:
            if array[0] > layer_amount:
                layer_amount = array[0]
        layer_amount = int(layer_amount)

        arrays_separated_by_layer = [[] for _ in range(layer_amount)]

        # make groups of layer_amount arrays
        parsed_arrays_by_layer_count = [
            parsed_arrays[n : n + layer_amount]
            for n in range(0, len(parsed_arrays), layer_amount)
        ]

        # regroup them into layer_amount groups of arrays
        for grouped_arrays in parsed_arrays_by_layer_count:
            for i in range(layer_amount):
                arrays_separated_by_layer[i].append(grouped_arrays[i])

        return arrays_separated_by_layer

    def __calculate_deltas(
        self, list_of_arrays: List[np.ndarray], mode: str
    ) -> List[np.ndarray]:
        """Calculate deltas between successive arrays."""
        deltas = []
        # the number of values for neuron data
        if mode == "p":
            neuron_data_length = 5
        else:
            neuron_data_length = 4
        previous_array = list_of_arrays[0]
        for current_array in list_of_arrays:
            delta = []
            for index, value in enumerate(current_array):
                # don't include the layer number
                if index != 0:
                    # don't delta the neuron id
                    if not (index - 1) % neuron_data_length == 0:
                        delta.append(previous_array[index] - value)
                    else:
                        delta.append(value)
            deltas.append(delta)
            previous_array = current_array
        return deltas

    def __zipped_delta_to_json(
        self, zipped_delta_array, num_layers, mode
    ) -> list[dict]:
        """Takes in a zipped delta array, the number of layers, and a mode. Returns a json of the training values."""
        json_output = []
        l = 0
        layers = [[] for _ in range(num_layers)]

        for t, delta in enumerate(zipped_delta_array):
            for i, _ in enumerate(delta):
                # Pyramidal mode
                if mode == "p" and i % 5 == 0:
                    layer = {
                        "id": delta[i],
                        "apical": delta[i + 1],
                        "basal": delta[i + 2],
                        "soma": delta[i + 3],
                        "firing": int(delta[i + 3] > delta[i + 4]),
                    }
                # Interneuronal mode
                elif mode == "i" and i % 4 == 0:
                    layer = {
                        "id": delta[i],
                        "dend": delta[i + 1],
                        "soma": delta[i + 2],
                        "firing": int(delta[i + 2] > delta[i + 3]),
                    }
                else:
                    continue
                layers[l].append(layer)

            # increment l
            l = (l + 1) % num_layers
            # if all layers are full
            if l == num_layers - 1:
                json_output.append({"time": ((t) // 3), "layers": layers})
                layers = [[] for _ in range(num_layers)]

        return json_output

    def __zip_delta_array(self, num_layers, delta_array):
        """Groups an array of deltas into groups of `num_layers` and returns it."""
        zipped_delta_array = []
        for i in range(len(delta_array[0]) * num_layers):
            zipped_delta_array.append(delta_array[i % num_layers][i // num_layers])

        return zipped_delta_array

    def __list_of_arrays_to_delta(
        self, list_of_arrays: List[np.ndarray], file_name: str, mode: str
    ) -> None:
        """Convert list of arrays into delta values and save in JSON format."""
        # get array delta values
        delta_array = []
        for array in list_of_arrays:
            delta_array.append(self.__calculate_deltas(array, mode))

        # get the number of layers
        num_layers = len(list_of_arrays)

        # zip the array of deltas
        zipped_delta_array, num_layers = self.__zip_delta_array(num_layers, delta_array)

        # get the output values
        json_output = self.__zipped_delta_to_json(zipped_delta_array, num_layers, mode)

        json_file = file_name.replace(".txt", "_Delta.json")
        with open(json_file, "w") as file:
            json.dump(json_output, file, indent=1)

    def file_to_delta(self, file_name: str, mode: str) -> None:
        """
        Takes in a file name and a mode ("p" for pyramidal, "i" for interneuron).
        Converts it to a `json` of training data.
        """
        list_of_arrays = self.__file_to_list_of_arrays(file_name)
        self.__list_of_arrays_to_delta(list_of_arrays, file_name, mode)
