from bfio import BioReader
import filepattern as fp
import os
import tensorstore as ts
import numpy as np
import concurrent.futures
from multiprocessing import get_context
import json
from typing import List, Tuple, Dict, Any



class VolumeGenerator:
    """
    A class for generating volumetric data from image files and storing it in a Zarr array.

    This class handles the process of reading image files, grouping them based on specified criteria,
    and writing the data into a Zarr array for efficient storage and access of large-scale volumetric data.

    Attributes:
        _source_dir (str): Directory containing the source image files.
        _group_by (str): Criterion for grouping images ('c', 't', or 'z').
        _file_pattern (str): Pattern to match the image files.
        _out_dir (str): Output directory for the generated Zarr array.
        _image_name (str): Name of the output Zarr array.
        _X (int): Width of the images.
        _Y (int): Height of the images.
        _Z (int): Depth of the volume.
        _T (int): Number of time points.
        _C (int): Number of channels.
        _x_unit (str): Unit of measurement for X dimension.
        _y_unit (str): Unit of measurement for Y dimension.
        _z_unit (str): Unit of measurement for Z dimension.
        files (List[str]): List of image file paths.
        _zarr_spec (Dict[str, Any]): Specification for the Zarr array.
        _base_scale_key (int): Base scale key for the Zarr array.
    """    
    _source_dir: str
    _group_by: str
    _file_pattern: str
    _out_dir: str
    _image_name: str
    _X: int
    _Y: int
    _Z: int
    _T: int
    _C: int
    _x_unit: str
    _y_unit: str
    _z_unit: str
    files: List[str]
    _zarr_spec: Dict[str, Any]
    _base_scale_key: int

    VALID_GROUP_BY = {'c', 't', 'z'}  # Define allowed values
    CHUNK_SIZE = 1024
    
    def __init__(
        self,
        source_dir: str,
        group_by: str,
        file_pattern: str,
        out_dir: str,
        image_name: str,
        base_scale_key: int = 0
    ) -> None:
        """
        Initialize the VolumeGenerator.

        Args:
            source_dir (str): Directory containing the source image files.
            group_by (str): Criterion for grouping images ('c', 't', or 'z').
            file_pattern (str): Pattern to match the image files.
            out_dir (str): Output directory for the generated Zarr array.
            image_name (str): Name of the output Zarr array.
            base_scale_key (int, optional): Base scale key for the Zarr array. Defaults to 0.
        """
        if group_by not in self.VALID_GROUP_BY:
            raise ValueError(f"group_by must be one of {self.VALID_GROUP_BY}")
        
        self._source_dir = source_dir
        self._group_by = group_by
        self._file_pattern = file_pattern
        self._out_dir = out_dir
        self._image_name = image_name
        self._base_scale_key = base_scale_key        

    def init_base_zarr_file(self):
        """
        Initialize the base Zarr file by determining dimensions and setting up the Zarr specification.

        This method analyzes the source files, determines the dimensions of the Zarr array,
        and sets up the Zarr specification for writing the data.
        """
        # find out what are the dimension of the zarr array
        fps = fp.FilePattern(self._source_dir, self._file_pattern)
        groups = [fi[0] for fi, _ in fps(group_by=self._group_by)]
        dimensions = [v for t in groups for v in t if isinstance(v, int)]
        # Get the number of layers to stack
        dim_size = len(dimensions)
        self.files = []
        for file in fps():
            self.files.append(file[1][0])
        # open a test file
        with BioReader(self.files[0], backend="tensorstore") as br:
            self._X = br.X
            self._Y = br.Y
            dtype = br.dtype
            if br.physical_size_x[1] is not None:
                self._x_unit = br.physical_size_x[1]
            else:
                self._x_unit = "micrometer"
            if br.physical_size_y[1] is not None:
                self._y_unit = br.physical_size_y[1]
            else:
                self._y_unit = "micrometer"
            if br.physical_size_z[1] is not None:
                self._z_unit = br.physical_size_z[1]
            else:
                self._z_unit = "micrometer"

        self._Z = 1
        self._T = 1
        self._C = 1

        if self._group_by == "c":
            self._C = dim_size
        elif self._group_by == "t":
            self._T = dim_size
        elif self._group_by == "z":
            self._Z = dim_size
        else:
            pass

        self._zarr_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": f"{self._out_dir}/{self._image_name}/{self._base_scale_key}",
            },
            "create": True,
            "delete_existing": False,
            "open": True,
            "metadata": {
                "shape": [self._C, self._Z, self._Y, self._X],
                "chunks": [1, 1, self.CHUNK_SIZE, self.CHUNK_SIZE],
                "dtype": np.dtype(dtype).str,
                "dimension_separator": "/",
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 1,
                    "shuffle": 1,
                    "blocksize": 0,
                },
            },
            "context": {
                "cache_pool": {},
                "data_copy_concurrency": {"limit": os.cpu_count() // 2},
                "file_io_concurrency": {"limit": os.cpu_count() // 2},
                "file_io_sync": False,
            },
        }
    def layer_writer(self, args: Tuple[str, Dict[str, Any], int, int]) -> None:
        """
        Write a single layer of the volume to the Zarr array.

        Args:
            args (Tuple[str, dict, int, int]): Tuple containing input file path, Zarr specification, z-index, and c-index.
        """
        input_file: str = args[0]
        zarr_spec: dict = args[1]
        z: int = args[2]
        c: int = args[3]

        zarr_array: ts.TensorStore = ts.open(zarr_spec).result()
        commit_futures: List[ts.Future[None]] = []
        try:
            br: BioReader = BioReader(input_file, backend="tensorstore")
            for y in range(0, br.Y, self.CHUNK_SIZE):
                y_max: int = min([br.Y, y + self.CHUNK_SIZE])
                for x in range(0, br.X, self.CHUNK_SIZE):
                    x_max: int = min([br.X, x + self.CHUNK_SIZE])
                    write_future = zarr_array[c, z, y:y_max, x:x_max].write(
                            br[y:y_max, x:x_max, 0, 0, 0].squeeze()
                        )
                    commit_futures.append(write_future.commit_future)

            for commit_future in commit_futures:
                commit_future.result()
        except Exception as e:
            print(f"Caught an exception for item : {e}")

    def generate_volume(self):
        """
        Generate the complete volume by initializing the Zarr file and writing the image stack.
        """
        self.init_base_zarr_file()
        self.write_image_stack()
        self._create_zattr_file()
        self._create_zgroup_file()

    def write_image_stack(self):
        """
        Write the entire image stack to the Zarr array using parallel processing.
        """
        count = 0
        arg_list = []
        for file_name in self.files:
            t = 0
            c = 0
            z = 0
            if self._group_by == "c":
                c = count
            elif self._group_by == "t":
                t = count
            elif self._group_by == "z":
                z = count
            else:
                pass

            arg_list.append((file_name, self._zarr_spec, z, c, t))
            count += 1
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count() // 2, mp_context=get_context("spawn")
        ) as executor:
            executor.map(self.layer_writer, arg_list)

    def _get_default_axes_metadata(self) -> List[dict]:
        """
        Generate the default axes metadata for the Zarr array.
        Returns:
            List[dict]: List of dictionaries representing the axes metadata.
        """
        axes_metadata: List[dict] = [
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ]
        return axes_metadata

    def _create_zgroup_file(self) -> None:
        """
        Creates a .zgroup file for the zarr pyramid.
        """
        zgroup_dict: dict = {"zarr_format": 2}

        with open(f"{self._out_dir}/{self._image_name}/.zgroup", "w") as json_file:
            json.dump(zgroup_dict, json_file)

        with open(f"{self._out_dir}/{self._image_name}/{self._base_scale_key}/.zgroup", "w") as json_file:
            json.dump(zgroup_dict, json_file)

    def _create_zattr_file(self) -> None:
        """
        Creates a .zattrs file for the zarr pyramid.
        """
        attr_dict: dict = {}
        
        if os.path.exists(f"{self._out_dir}/{self._image_name}/.zattrs"):
            with open(f"{self._out_dir}/{self._image_name}/.zattrs", "r") as json_file:
                attr_dict_base = json.load(json_file)         
                if "axes" in attr_dict_base:
                    axes_metadata = attr_dict_base["axes"]
                else:
                    axes_metadata = self._get_default_axes_metadata()
        else:
            axes_metadata = self._get_default_axes_metadata()

        attr_dict["axes"] = axes_metadata
        multiscale_metadata: list = [
            {
            "coordinateTransformations": [
                {"scale": [1, 1, 1, 1], "type": "scale"}
            ],
            "path": f"{self._base_scale_key}",
            }
        ]

        attr_dict["datasets"] = multiscale_metadata
        attr_dict["version"] = "0.4"
        attr_dict["name"] = self._image_name
        attr_dict["metadata"] = {"method": "mean"}

        final_attr_dict = {"multiscales": [attr_dict]}

        with open(f"{self._out_dir}/{self._image_name}/.zattrs", "w") as json_file:
            json.dump(final_attr_dict, json_file)

class PyramidGenerator3D:
    """
    A class for generating multi-resolution pyramids from Zarr arrays.

    This class handles the creation of downsampled versions of a base Zarr array,
    forming a multi-resolution pyramid for efficient data access at different scales.

    Attributes:
        _zarr_loc_dir (str): Directory containing the base Zarr array.
        _base_scale_key (int): Key of the base scale in the Zarr array.
        _image_name (str): Name of the image derived from the Zarr directory.
    """

    _zarr_loc_dir: str
    _base_scale_key: int
    _image_name: str
    
    CHUNK_SIZE = 1024

    def __init__(self, zarr_loc_dir, base_scale_key):
        """
        Initialize the PyramidGenerator3D.

        Args:
            zarr_loc_dir (str): Directory containing the base Zarr array.
            base_scale_key (int): Key of the base scale in the Zarr array.
        """
        self._zarr_loc_dir = zarr_loc_dir
        self._base_scale_key = base_scale_key

        self._image_name = os.path.basename(self._zarr_loc_dir)

    def downsample_pyramid(self, level: int) -> None:
        """
        Downsamples the pyramid at the specified level.

        This method creates a downsampled version of the base image at the given pyramid level.
        It uses the TensorStore library to read the base image, downsample it, and write the
        result to a new Zarr array.

        Args:
            level (int): The pyramid level to generate. This determines the downsampling factor,
                         which is 2^level in each dimension.

        The method performs the following steps:
        1. Sets up the downsampling specification using TensorStore.
        2. Opens the downsampled array for reading.
        3. Creates a new Zarr array specification for writing the downsampled data.
        4. Opens the new Zarr array for writing.
        5. Reads chunks from the downsampled array and writes them to the new Zarr array.

        The downsampled array is written in chunks to optimize performance and memory usage.
        """
        ds_spec: Dict[str, Any] = {
            "driver": "downsample",
            "downsample_factors": [1, 2**level, 2**level, 2**level],
            "downsample_method": "mean",
            "base": {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": f"{self._zarr_loc_dir}/{self._base_scale_key}",
                },
            },
        }

        ds_zarr_array: ts.TensorStore = ts.open(ds_spec).result()
        [C, Z, Y, X] = ds_zarr_array.shape
        ds_write_spec: Dict[str, Any] = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": f"{self._zarr_loc_dir}/{level}",
            },
            "create": True,
            "delete_existing": True,
            "metadata": {
                "shape": ds_zarr_array.shape,
                "chunks": [1, 1, self.CHUNK_SIZE, self.CHUNK_SIZE],
                "dtype": np.dtype(ds_zarr_array.dtype.numpy_dtype).str,
                "dimension_separator": "/",
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 1,
                    "shuffle": 1,
                    "blocksize": 0,
                },
            },
        }

        ds_zarr_array_write: ts.TensorStore = ts.open(ds_write_spec).result()

        commit_futures: List[ts.Future[None]] = []

        for c in range(C):
            for z in range(Z):
                for y in range(0, Y, self.CHUNK_SIZE):
                    y_max = min([Y, y + self.CHUNK_SIZE])
                    for x in range(0, X, self.CHUNK_SIZE):
                        x_max = min([X, x + self.CHUNK_SIZE])
                        block = ds_zarr_array[c, z, y:y_max, x:x_max].read().result()
                        write_future = ds_zarr_array_write[c, z, y:y_max, x:x_max].write(block)
                        commit_futures.append(write_future.commit_future)

        # Ensure all writes are durable (i.e., non-transactional and committed)
        for commit_future in commit_futures:
            commit_future.result()

        self._create_zgroup_file(f"{self._zarr_loc_dir}/{level}")

    def _create_zgroup_file(self, file_loc: str) -> None:
        """
        Creates a .zgroup file for the zarr pyramid.
        """
        zgroup_dict: dict = {"zarr_format": 2}

        with open(f"{file_loc}/.zgroup", "w") as json_file:
            json.dump(zgroup_dict, json_file)

    def _create_zattr_file(self, num_levels: int) -> None:        
        """
        Creates a .zattrs file for the zarr pyramid.
        """
        attr_dict: dict = {}

        # add logic to constrain num_levels

        axes_metadata = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        attr_dict["axes"] = axes_metadata
        multiscale_metadata: list = []
        downsample_factor = 1
        base_scale_metadata = {
            "coordinateTransformations": [
                {"scale": [1, downsample_factor, downsample_factor, downsample_factor], "type": "scale"}
            ],
            "path": f"{self._base_scale_key}",
        }

        multiscale_metadata.append(base_scale_metadata)

        for key in range(1, num_levels+1):
            downsample_factor *= 2
            metadata = {
                "coordinateTransformations": [
                    {"scale": [1, downsample_factor, downsample_factor, downsample_factor], "type": "scale"}
                ],
                "path": f"{key}",
            }
            multiscale_metadata.append(metadata)

        attr_dict["datasets"] = multiscale_metadata
        attr_dict["version"] = "0.4"
        attr_dict["name"] = self._image_name
        attr_dict["metadata"] = {"method": "mean"}

        final_attr_dict = {"multiscales": [attr_dict]}

        with open(f"{self._zarr_loc_dir}/.zattrs", "w") as json_file:
            json.dump(final_attr_dict, json_file)

    def generate_pyramid(self, num_levels: int) -> None:
        """
        Generate the multi-resolution pyramid.

        This method creates the .zattrs file for the pyramid and generates
        downsampled versions of the base image for each specified level.

        Args:
            num_levels (int): Number of pyramid levels to generate.
        """
        self._create_zattr_file(num_levels)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count() // 2, mp_context=get_context("spawn")
        ) as executor:
            executor.map(self.downsample_pyramid, range(1, num_levels+1))
