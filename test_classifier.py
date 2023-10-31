import bottle

import image_util
from nsfw_classifier import NsfwClassifier
from face_detector import FaceDetector
import classifier_webapi
import time
import webtest


class TestClassifier:
    nsfw_classifier = NsfwClassifier()
    face_detector = FaceDetector()

    def _local(self, image):
        t0 = time.time_ns() / 1000000
        image = image_util.load_image(image)
        t1 = time.time_ns() / 1000000
        nsfw = self.nsfw_classifier.classify(image)[0]
        t2 = time.time_ns() / 1000000
        faces = self.face_detector.detect(image)[0]
        t3 = time.time_ns() / 1000000
        print("")
        print("Image loaded in {:0.0f} ms".format(t1 - t0))
        print("NSFW classified in {:0.0f} ms".format(t2 - t1))
        print("Faces detected in {:0.0f} ms".format(t3 - t2))
        print("NSFW: {}".format(nsfw))
        print("Faces: {}".format(faces))
        return nsfw, faces

    def test_local_sfw_noface(self):
        (nsfw, faces) = self._local('images/test1.jpg')
        assert nsfw['sfw']
        assert nsfw['sfw_definite']
        assert len(faces) == 0

    # def test_local_sfw_oneface(self):
    #     (nsfw, faces) = self._local('images/test2.jpg')
    #     assert nsfw['sfw']
    #     assert nsfw['sfw_definite']
    #     assert len(faces) == 1

    # def test_local_sfw_twofaces(self):
    #     (nsfw, faces) = self._local('images/test3.jpg')
    #     assert nsfw['sfw']
    #     assert nsfw['sfw_definite']
    #     assert len(faces) == 2

    def test_server(self):
        app = classifier_webapi.get_app(self.nsfw_classifier, self.face_detector)
        bottle.debug(True)
        test_app = webtest.TestApp(app)
        response = test_app.post("/ica/v1/classify", upload_files=[
            ('image', "images/test1.jpg")
        ])
        assert response.json['sfw']
        assert response.json['sfw_definite']
        assert len(response.json['faces']) == 0
