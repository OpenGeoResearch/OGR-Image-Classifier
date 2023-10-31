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
import bottle
import io
import json

import image_util
from nsfw_classifier import NsfwClassifier
from face_detector import FaceDetector


@bottle.route('/ica/v1/classify', method='POST')
def classify():
    global nsfw_classifier, face_detector
    bottle.response.content_type = 'application/json'
    images_fileupload = bottle.request.files.getall('image')
    if (len(images_fileupload)) == 0:
        return bottle.HTTPResponse({'error': 'No images provided.'}, 400)
    if (len(images_fileupload)) > 1:
        return bottle.HTTPResponse({'error': 'Must not provide more than one image.'}, 400)
    # Download the image into an in-memory BytesIO object
    image = io.BytesIO()
    images_fileupload[0].save(image)
    # Load the image in-memory and classify it
    image = image_util.load_image(image)
    nsfw_result = nsfw_classifier.classify(image)[0]
    face_result = face_detector.detect(image)[0]
    return json.dumps({**nsfw_result, 'faces': face_result}, indent=4)


def get_app(nsfw_classifier_param, face_detector_param):
    global nsfw_classifier, face_detector
    assert isinstance(nsfw_classifier_param, NsfwClassifier)
    assert isinstance(face_detector_param, FaceDetector)
    nsfw_classifier = nsfw_classifier_param
    face_detector = face_detector_param
    return bottle.default_app()


def start_server(nsfw_classifier_param, face_detector_param):
    bottle.run(app=get_app(nsfw_classifier_param, face_detector_param), host="0.0.0.0", port=9000, server="waitress")
