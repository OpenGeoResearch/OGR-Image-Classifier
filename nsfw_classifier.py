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
import numpy
import tensorflow
import tensorflow_hub
import image_util

class NsfwClassifier:
    # If any NSFW category (porn, sexy, ...) is above nsfw_threshold, mark the image as NSFW
    nsfw_threshold = 0.9
    # If an NSFW category is above nsfw_definite_threshold, mark the image as definite NSFW
    nsfw_definite_threshold = 0.98
    # If an image was not marked as NSFW, and one of the safe categories (normal, drawing) is above this threshold,
    # mark it as definite SFW
    sfw_definite_threshold = 0.9

    def __init__(self, model_path="models/nsfw_mobilenet2.224x224.h5", model_image_dim=(224, 224)):
        self.model = tensorflow.keras.models.load_model(model_path,
                                                        custom_objects={'KerasLayer': tensorflow_hub.KerasLayer})
        self.model_image_dim = model_image_dim

    def __prepare_image(self, image):
        image = tensorflow.image.resize(image, self.model_image_dim, method='bicubic')
        image /= 255
        return image

    def classify(self, images):
        """
        Performs SFW/NSFW classification on a list of images.

        For example, inputting one SFW image may produce the following result:

        [
            {
                'sfw': True,
                'sfw_definite': True,
                'sfw_raw': {
                    'drawings': 0.000446719495812431,
                    'hentai': 0.0002857570652849972,
                    'neutral': 0.987395703792572,
                    'porn': 0.01047840528190136,
                    'sexy': 0.0013934546150267124
                }
            }
        ]

        :param images: A list of images to detect faces in.
            These can be `numpy.ndarray`, in which case they will be assumed to be RGB images.
            If this is not the case for an image, it is first passed to `image_util.load_image` to load it into an
            ndarray.
            If this is parameter is not a list, it will be wrapped in one, resulting in a list of length 1.
        :return: A list of the same length as the parameter image. For every image in images, this list contains
            information about the SFW / NSFW classification result.
        """
        # Ensure that images is a list
        # If we're handed a single image, put it in a list
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            raise ValueError("Must provide at least one image")

        images_loaded = []
        for image in images:
            if not isinstance(image, numpy.ndarray):
                image = image_util.load_image(image)
            images_loaded.append(self.__prepare_image(image))

        model_predictions = self.model.predict(numpy.asarray(images_loaded))
        predictions = []
        for model_prediction in model_predictions:
            drawings = float(model_prediction[0])
            hentai = float(model_prediction[1])
            neutral = float(model_prediction[2])
            porn = float(model_prediction[3])
            sexy = float(model_prediction[4])
            sfw = True
            definite = False
            for nsfw_prediction in [hentai, porn, sexy]:
                if nsfw_prediction > self.nsfw_threshold:
                    sfw = False
                    if nsfw_prediction > self.nsfw_definite_threshold:
                        definite = True
            if sfw:
                for sfw_prediction in [drawings, neutral]:
                    if sfw_prediction > self.sfw_definite_threshold:
                        definite = True
            predictions.append({
                'sfw': sfw,
                'sfw_definite': definite,
                'sfw_raw': {
                    'drawings': drawings,
                    'hentai': hentai,
                    'neutral': neutral,
                    'porn': porn,
                    'sexy': sexy
                }
            })
        return predictions
