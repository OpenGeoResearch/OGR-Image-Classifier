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
import functools

import numpy
import image_util
import retinaface.RetinaFace
import retinaface.model.retinaface_model


class FaceDetector:
    def __init__(self, model_path: "Filesystem path for model weights" = 'models/retinaface.h5'):
        # By default, the RetinaFace library downloads the network weights from a GitHub repository and stores them
        # in a subdirectory of the current user's home folder.
        # This is bad:
        # - Increases startup time, especially in single-use containers
        # - No automated cleanup, pollutes user directory
        # - Breaks the app if the third-party GitHub repo goes offline
        # Unfortunately, RetinaFace does not provide an API to load the weights from a specified file. Thus, we hook /
        # replace the function that loads the weights with one that loads them from our specified location.
        retinaface.model.retinaface_model.load_weights = functools.partial(FaceDetector.__hook_load_weights, model_path)
        retinaface.RetinaFace.build_model()

    @staticmethod
    def __hook_load_weights(path, model):
        model.load_weights(path)
        return model

    def detect(self, images: "(list of) image(s) to detect faces in"):
        """
        Performs face detection on a list of images.

        For example, inputting one image with one face in it may produce the following result:

        [
            [{'score': 0.999523401260376, 'facial_area': {'min_x': 206, 'min_y': 65, 'max_x': 327, 'max_y': 231}}]
        ]

        :param images: A list of images to detect faces in.
            These can be `numpy.ndarray`, in which case they will be assumed to be RGB images.
            If this is not the case for an image, it is first passed to `image_util.load_image` to load it into an
            ndarray.
            If this is parameter is not a list, it will be wrapped in one, resulting in a list of length 1.
        :return: A list of the same length as the parameter image. For every image in images, this list contains a list
            of all faces detected in that image.
        """
        # Ensure that images is a list
        # If we're handed a single image, put it in a list
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            raise ValueError("Must provide at least one image")

        predictions = []
        for image in images:
            if not isinstance(image, numpy.ndarray):
                image = image_util.load_image(image)
            # This one weird line converts an RGB image to BGR
            # RetinaFace uses cv2.imread(...) by default, which loads them as BGR, which the network is trained on.
            # However, PIL / image_util.load_image(...) loads as RGB, which necessitates the conversion.
            image = image[..., ::-1].copy()
            raw_faces = retinaface.RetinaFace.detect_faces(image)
            faces = []
            # For some reason, RetinaFace may output a weird tuple if an image contains no faces. Thus, we need to check
            # the type before iterating over it.
            if isinstance(raw_faces, dict):
                for face_name, face_data in raw_faces.items():
                    faces.append({
                        'score': float(face_data['score']),
                        'facial_area': {
                            'min_x': float(face_data['facial_area'][0]),
                            'min_y': float(face_data['facial_area'][1]),
                            'max_x': float(face_data['facial_area'][2]),
                            'max_y': float(face_data['facial_area'][3])
                        }
                    })
            predictions.append(faces)
        return predictions
