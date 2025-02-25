from pathlib import Path
import json
import os
import math
import shutil

import numpy as np
import ome_types
import tensorstore as ts

from .libargolid import PyramidCompositorCPP


class PyramidCompositor:
    """
    A class for composing a group of pyramid images into an assembled pyramid structure.
    """

    def __init__(
        self, input_pyramids_loc: str, out_dir: str, output_pyramid_name: str
    ) -> None:
        """
        Initializes the PyramidCompositor object.

        Args:
            input_pyramids_loc (str): The location of the input pyramid images.
            out_dir (str): The output directory for the composed zarr pyramid file.
            output_pyramid_name (str): The name of the zarr pyramid file.
        """
        self._pyramid_compositor_cpp = PyramidCompositorCPP(input_pyramids_loc, out_dir, output_pyramid_name)

    def _create_xml(self) -> None:
        """
        Writes an OME-XML metadata file for the pyramid.
        """
        self._pyramid_compositor_cpp.create_xml()
        

    def _create_zattr_file(self) -> None:
        """
        Creates a .zattrs file for the zarr pyramid.
        """
        self._pyramid_compositor_cpp.create_zattr_file()

    def _create_zgroup_file(self) -> None:
        """
        Creates .zgroup files for the zarr pyramid.
        """
        self._pyramid_compositor_cpp.create_zgroup_file()

    def _create_auxilary_files(self) -> None:
        """
        Creates auxiliary files (OME-XML, .zattrs, .zgroup) for the pyramid.
        """
        self._pyramid_compositor_cpp.create_auxilary_files()
        

    def _write_zarr_chunk(
        self, level: int, channel: int, y_index: int, x_index: int
    ) -> None:
        """
        Writes the chunk file at the specified level, channel, y_index, and x_index.

        Args:
            level (int): The level of the pyramid.
            channel (int): The channel of the pyramid.
            y_index (int): The y-index of the tile.
            x_index (int): The x-index of the tile.
        """
        self._pyramid_compositor_cpp.write_zarr_chunk(level, channel, y_index, x_index)

    def set_composition(self, composition_map: dict) -> None:
        """
        Sets the composition for the pyramid.

        Args:
            composition_map (dict): A dictionary mapping composition images to file paths.
        """
        self._pyramid_compositor_cpp.set_composition(composition_map)

    def reset_composition(self) -> None:
        """
        Resets the pyramid composition by removing the pyramid file and clearing internal data structures.
        """
        self._pyramid_compositor_cpp.reset_composition()

    def get_zarr_chunk(
        self, level: int, channel: int, y_index: int, x_index: int
    ) -> None:
        """
        Retrieves zarr chunk data from the pyramid at the specified level, channel, y_index, and x_index.

        This method checks if the requested zarr chunk is already in the cache. If not, it calls
        the `_write_zarr_chunk` method to generate the chunk and add it to the cache.

        Args:
            level (int): The level of the pyramid.
            channel (int): The channel of the pyramid.
            y_index (int): The y-index of the tile.
            x_index (int): The x-index of the tile.

        Raises:
            ValueError: If the composition map is not set, the requested level does not exist,
                the requested channel does not exist, or the requested y_index or x_index
                is out of bounds.
        """

        self._pyramid_compositor_cpp.write_zarr_chunk(level, channel, y_index, x_index)

