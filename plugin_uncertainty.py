import instantngp.build.pyngp as ngp_nerftools_interface
from io import TextIOWrapper
import logging
from radiancefield import IRadianceField
import numpy as np

class RadianceField(IRadianceField):
    
    def __init__(
        self,
        batch_size: int,
        trained_network_weights_file: str,
        apply_output_activation: bool = True,
        samples_per_pixels: int = 16,
        background_color: list = [0.0, 0.0, 0.0, 1.0]
    ):

        super().__init__(
            batch_size,
            apply_output_activation
        )

        self.name = "InstantNGP"
        self._trained_network_weights_file = trained_network_weights_file
        self._samples_per_pixels = samples_per_pixels
        self._background_color = background_color

        self._interface = ngp_nerftools_interface.NeRFToolsInterface(ngp.TestbedMode.Nerf)
        self._interface.load_snapshot(trained_network_weights_file)
        self._interface.background_color = background_color

        self._dataset_scale = self._interface.get_dataset_scale()
        self._dataset_offset = self._interface.get_dataset_offset()

        logging.info('NERFTOOLS.PLUGINS.INSTANT-NGP: initialized')



    def convert_coordinates(self, coordinates: np.array) -> np.array:
        n = coordinates.shape[0] 
        converted_coordinates = np.zeros((n, 3))
    
        for i in range(n):
            coordinate = coordinates[i]
            converted_coordinate = self._dataset_scale * np.array([coordinate[1], coordinate[2], coordinate[0]]) + self._dataset_offset
            converted_coordinates[i] = converted_coordinate
    
        return converted_coordinates



    def _get(
        self,
        positions : np.array, # n * m * 3 (float, n: rays, m: points on ray)
        directions : np.array = None # n * m * 3 (float, n: rays, m: points on ray)
    ) -> np.array:  # n * m * 4 (float, n: rays, m: points on ray, 0-2: color, 3: density)

        network_output = self._interface.inference(
            self._apply_output_activation,
            self._reshape(positions),
            self._reshape(directions)
        )

        return np.reshape(
            network_output,
            (
                positions.shape[0],
                positions.shape[1],
                4
            )
        )
    
    def _reshape(
        self,
        values: np.array # n * m * 3  
    ) -> np.array: # k * 3 (k = n * m)

        return np.reshape(
            values,
            (
                values.shape[0] * values.shape[1], 
                3
            )
        )