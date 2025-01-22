import sys
import string
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
from plugin_uncertainty import RadianceField as InstantNGPRadianceField
from tqdm import tqdm
import struct
from concurrent.futures import ThreadPoolExecutor



def create_member_grid(coordinate_list, filename, radiance_field, ply_writer, density_threshold=0, use_threshold=False, batch_size=100):
    logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: getting densities and radiances... SAFE AS ARRAY.:")
    
    logging.info(f"convert coordinate")
    coord_batches = [np.array(coordinate_list[i:i+batch_size]) for i in range(0, len(coordinate_list), batch_size)]
    coord_converted_batches = [radiance_field.convert_coordinates(batch) for batch in coord_batches]
    logging.info(f"get values")
    values_batches = [radiance_field.get(batch[:, None, :]) for batch in coord_converted_batches]
    
    logging.info(f"get densities")
    densities = np.concatenate([values[:, :, 3].flatten() for values in values_batches])
    logging.info(f"get radiances")
    radiances = np.concatenate([values[:, :, :3].reshape((-1, 3)) for values in values_batches])
    
    Grid_Array = np.hstack((coordinate_list, densities[:, None], radiances)).astype(np.float32)   #hier auf 32 wegen inf werte bei std? davor 16 wegen Platz

    logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: Array size {len(Grid_Array)}")
    return Grid_Array
    
    
    
    
def create_coordinate_list(                               
    min_coordinate: np.array, 
    max_coordinate: np.array, 
    resolution: np.float16 = 1.0,

):
    grid_size = np.ceil((max_coordinate - min_coordinate) / resolution)
    grid_size = grid_size.astype(int) + 1

    grid_0 = np.array([
                min_coordinate + np.array([ 0, y, z ], dtype=np.float16) * resolution
            for z in range(grid_size[2])
        for y in range(grid_size[1])
    ])

    coordinate_list = zarr.creation.array(grid_0,dtype=np.float16)
    print('coordinate_list.size')
    print(coordinate_list.size)

    logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: creating point list...")

    for x in tqdm(range(1,grid_size[0])):        
        grid_0[:,0] = min_coordinate[0] + x * resolution
        
        coordinate_list.append(grid_0)      

    print('single coordinate')
    print(coordinate_list[1])
    logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: calculated {len(coordinate_list[:])} points")

    return coordinate_list

def write_PC_(data,filename):
    logging.info(f"Punkt 0 {data[0]}")
    logging.info(f"Punkt 1 {data[1]}")
    logging.info(f"Punkt 2 {data[2]}")
    logging.info(f"Punkt 3 {data[3]}")
    logging.info(f"Punkt 4 {data[4]}")
    logging.info(f"Punkt 5 {data[5]}")
    logging.info(f"Punkt 0 {data[6]}")
    logging.info(f"Punkt 1 {data[7]}")
    logging.info(f"Punkt 2 {data[8]}")
    logging.info(f"Punkt 3 {data[9]}")
    logging.info(f"Punkt 4 {data[10]}")
    logging.info(f"Punkt 5 {data[11]}")
    logging.info(f"Anzahl Punkte {len(data)}")
    logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: Speichere Punktwolke als Binary...")
    with open(filename, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(b'element vertex ' + str(len(data)).encode() + b'\n')
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float density\n')
        f.write(b'property float red\n')
        f.write(b'property float green\n')
        f.write(b'property float blue\n')
        f.write(b'end_header\n')

        for item in tqdm(data):
            #logging.info(f"items {item[0], item[1], item[2], item[3], item[4], item[5], item[6]}")
            if not np.isnan(item[0]):
                binary = struct.pack('fffffff', *item)
                f.write(binary)

               
                
def write_PC_ensemble(data,filename):
    logging.info(f"Anzahl Punkte {len(data)}")
    logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: Speichere Punktwolke als Binary...")
    with open(filename, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(b'element vertex ' + str(len(data)).encode() + b'\n')
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float mean_density_percentil\n')
        f.write(b'property float std_density_percentil\n')
        f.write(b'property float mean_red\n')
        f.write(b'property float mean_green\n')
        f.write(b'property float mean_blue\n')
        f.write(b'end_header\n')

        for item in tqdm(data):
            if not np.isnan(item[0]):
                binary = struct.pack('ffffffff', *item)
                f.write(binary)                               
                


'--------------------------------Writer--------------------------------------------------------'

class PlyPointWriter:                                      # Writes pointcloud to .ply file            

    _POINT_COUNT_PLACEHOLDER = "<POINT COUNT>"
   
    def __init__(
        self,
        file_path: str,
        use_color: bool = False
    ):
        self._file_path = file_path
        self._use_color = use_color

        self._temp_file_path = get_file_path_with_postfix(
            file_path,
            "_temp"
        )

    def __enter__(
        self
    ):
        self.open()

        return self

    def __exit__(
        self, 
        type, 
        value, 
        traceback
    ):
        if value is not None:
            raise value

        self.close()

    def open(
        self
    ):
        self._point_count = 0
        self._temp_file = open(self._temp_file_path, "w")

        self._temp_file.write("ply\n")
        self._temp_file.write("format ascii 1.0\n")
        self._temp_file.write(f"element vertex {self._POINT_COUNT_PLACEHOLDER}\n")
        self._temp_file.write("property float x\n")
        self._temp_file.write("property float y\n")
        self._temp_file.write("property float z\n")
        self._temp_file.write("property float density\n")

        if self._use_color:
            self._temp_file.write("property float red\n")
            self._temp_file.write("property float green\n")
            self._temp_file.write("property float blue\n")

        self._temp_file.write("element face 0\n")
        self._temp_file.write("property list uchar int vertex_indices\n")
        self._temp_file.write("end_header\n")
    
    def write_point(
        self,
        point: np.array,
        density: np.float32,
        color: np.array = None
    ):
        self._point_count += 1

        self._temp_file.write(f"{point[0]} {point[1]} {point[2]} {density}")

        if self._use_color and color is not None:
            self._temp_file.write(f" {color[0]} {color[1]} {color[2]}")

        self._temp_file.write("\n")

    def close(
        self
    ):
        self._temp_file.close()

        with open(self._temp_file_path, "r") as temp_file:
            with open(self._file_path, "w") as file:
                logging.info(f"NERFTOOLS.POINTCLOUDSCRIPTE: writing ply file...")
                while True:
                    
                    line = temp_file.readline()

                    if len(line) == 0:
                        break

                    if self._POINT_COUNT_PLACEHOLDER in line:

                        line = line.replace(
                            self._POINT_COUNT_PLACEHOLDER,
                            str(self._point_count)
                        )
                    
                    file.write(line)
        
        os.remove(self._temp_file_path)

def get_file_path_with_postfix(
    file_path: str,
    postfix: str
):
    path = Path(file_path)
    return f"{path.parent}/{path.stem}{postfix}{path.suffix}"


