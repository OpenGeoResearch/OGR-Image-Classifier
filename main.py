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
from nsfw_classifier import NsfwClassifier
from face_detector import FaceDetector
import classifier_webapi

if __name__ == '__main__':
    classifier_webapi.start_server(NsfwClassifier(), FaceDetector())
