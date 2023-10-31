# Copyright (C) 2023 Geodätisches Institut RWTH Aachen University,
# Mies-van-der-Rohe-Straße 1, D 52074 Aachen, Germany.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import PIL.Image
import tensorflow


def load_image(image):
    """
    Loads an image (bytes or filesystem path) into a float32 RGB ndarray.
    """
    # PIL.Image.open(...) supports loading images in-memory, which
    # tensorflow.keras.preprocessing.image.load_img(...) as of writing (2.7.0) does not.
    image_loaded = PIL.Image.open(image)
    image_loaded = image_loaded.convert('RGB')
    image_loaded = tensorflow.keras.preprocessing.image.img_to_array(image_loaded, dtype="float32")
    return image_loaded
