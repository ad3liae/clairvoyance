import asyncio
import itertools
import random
import sys
import os
import functools
import math
import time
import logging

import cv2
import numpy as np
import skvideo.io
from lipnet.lipreading.videos import Video

from clairvoyance.core import Speaker

from keras import backend as K
from scipy import ndimage
from scipy.misc import imresize
import dlib

import face_recognition
from face_recognition.api import cnn_face_detector

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','..','LipNet','common','predictors','shape_predictor_68_face_landmarks.dat')

class FaceRecognitionTask:
    def __init__(self, config, q):
        self._config = config
        self._q = q
        self._log = logging.getLogger(self.__class__.__name__)

    async def do(self):
        for video_path in self._config.targets:
            dec = VideoDecoder(video_path)
            total = dec.num_blocks()
            for nr,block in dec.decoded_blocks():
                self._log.debug("Sending batch #{} (of {})".format(nr, total))
                began_at = time.time()
                faces = [x for x in FaceDetector(face_predictor_path=FACE_PREDICTOR_PATH, preview=self._config.show_frame, face_detector_type=self._config.face_detector, face_detect_subsample=self._config.face_detect_subsample, face_updates=self._config.face_updates).do(block, framerate=dec._framerate())]
                self._log.debug("Faces detected: {} ({:.02f} sec.).".format(len(faces), time.time() - began_at))
                for video in faces:
                    await asyncio.get_event_loop().run_in_executor(None, self._q.put, Speaker(video=video, identity='Speaker #0'))
        if self._config.show_frame:
            cv2.destroyAllWindows()

class VideoDecoder:
    def __init__(self, video_path):
        self._path = video_path
        self._blocksize = 75
        self._meta = skvideo.io.ffprobe(video_path)
        self._gen = skvideo.io.vreader(video_path)

    @functools.lru_cache(maxsize=1)
    def _framerate(self):
        frtxt = self._meta['video']['@r_frame_rate']
        s = frtxt.split('/')
        if len(s) > 1:
            return float(s[0])/float(s[1])
        else:
            return float(s[0])

    @functools.lru_cache(maxsize=1)
    def num_blocks(self):
        return math.ceil((float(self._meta['video']['@duration'])*self._framerate()) / self._blocksize)

    def decoded_blocks(self):
        for nr in range(self.num_blocks()):
            yield nr, np.array(list(itertools.islice(self._gen, self._blocksize)))

class FaceDetector:
    def __init__(self, face_predictor_path=None, preview=False, face_detector_type='hog', face_detect_subsample=2, face_updates=5):
        if face_predictor_path is None:
            raise AttributeError('Face video need to be accompanied with face predictor')
        self.face_predictor_path = face_predictor_path
        self.preview = preview
        self.framerate = None
        self.known_face_encodings = []
        self.known_face_names = []
        self._detector = FaceDetector.detector_of_type(face_detector_type)
        self._face_detect_subsample = face_detect_subsample
        self._face_updates = face_updates

    @staticmethod
    def detector_of_type(type_):
        if type_ == 'hog':
            return dlib.get_frontal_face_detector()
        elif type_ == 'cnn':
            return Video.cnn_face_detector

    def do(self, frames, framerate=25):
        self.framerate = framerate
        detector = self._detector
        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        v = Video(vtype='face', face_predictor_path=self.face_predictor_path)
        v.face = np.array(frames)
        v.mouth = np.array(mouth_frames)
        v.set_data(mouth_frames)
        yield v

    def get_frames_mouth(self, detector, predictor, frames):
        return self.mouth_frames_of_a_face(detector, predictor, frames)

    def get_frame_rate(self, path):
        frtxt = skvideo.io.ffprobe(path)['video']['@r_frame_rate']
        s = frtxt.split('/')
        if len(s) > 1:
            return float(s[0])/float(s[1])
        else:
            return float(s[0])

    def mouth_frames_of_a_face(self, detector, predictor, frames):
        frameskip = 0
        mouth_frames = []
        known_as = dict()
        for nr, frame in enumerate(frames):
            if frameskip:
                if frameskip > 10:
                    frameskip = 0
                else:
                    frameskip = max(0, frameskip - 1)
            began_at = time.time()
            if self.preview and not frameskip:
                showframe = frame.copy()
            if self._face_detect_subsample > 1:
                scale = 1.0 / self._face_detect_subsample
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            else:
                scale = 1
            dets = detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)

                if nr % self._face_updates == 0:
                    face_encoding = face_recognition.face_encodings(frame, [(shape.rect.left(), shape.rect.top(), shape.rect.right(), shape.rect.bottom())])[0]
                    if self.known_face_encodings:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] <= 0.6:
                            known_as[k] = self.known_face_names[best_match_index]

                    if k not in known_as:
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append("known_{}".format(len(self.known_face_names)))

                if self.preview and not frameskip:
                    scale = int(1 / scale)
                    cv2.rectangle(showframe, (scale*shape.rect.left(), scale*shape.rect.top()), (scale*shape.rect.right(), scale*shape.rect.bottom()), (255,0,0), 2)
                    cv2.rectangle(showframe, (scale*shape.rect.left(), scale*shape.rect.bottom() - 17), (scale*shape.rect.right(), scale*shape.rect.bottom()), (255, 0, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(showframe, known_as[k] if k in known_as else 'Unknown', (scale*shape.rect.left() + 3, scale*shape.rect.bottom() - 3), font, 0.5, (255, 255, 255), 1)

            if shape is None: # Detector doesn't detect face, interpolate with the last frame
                try:
                    mouth_frames.append(mouth_frames[-1])
                except IndexError:
                    # XXX
                    mouth_frames.append(np.zeros((50,100,3), dtype='uint8'))
            else:
                mouth_frames.append(self.mouth_frame_of_face_shaped(shape, frame))
            elapsed = time.time() - began_at
            if self.preview and not frameskip:
                deadline = 1000/self.framerate
                slack = int(deadline - elapsed*1000)
                if slack < 0:
                    frameskip = math.ceil(-slack / deadline)
                cv2.imshow('Video', cv2.cvtColor(showframe, cv2.COLOR_RGB2BGR))
                cv2.waitKey(max(1, slack))
        return mouth_frames

    def mouth_frame_of_face_shaped(self, shape, frame):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        i = -1

        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = imresize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        return imresize(resized_img[mouth_t:mouth_b, mouth_l:mouth_r], (100, 50))

    @staticmethod
    def cnn_face_detector(*args, **kwargs):
        return (x.rect for x in cnn_face_detector(*args, **kwargs))

    def get_video_frames(self, path):
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        return frames
