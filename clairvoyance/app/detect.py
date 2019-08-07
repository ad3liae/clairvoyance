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
                for name,video in faces:
                    await asyncio.get_event_loop().run_in_executor(None, self._q.put, Speaker(video=video, identity=name))
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
        self._face_predictor_path = face_predictor_path
        self._preview = preview
        self._framerate = None
        self._known_face_encodings = []
        self._known_face_names = []
        self._detector = FaceDetector.detector_of_type(face_detector_type)
        self._face_detect_subsample = face_detect_subsample
        self._face_updates = face_updates
        self._log = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def detector_of_type(type_):
        if type_ == 'hog':
            return dlib.get_frontal_face_detector()
        elif type_ == 'cnn':
            return FaceDetector._cnn_face_detector

    def do(self, frames, framerate=25):
        self._framerate = framerate
        detector = self._detector
        predictor = dlib.shape_predictor(self._face_predictor_path)
        for name, mouth_frames in self._mouth_frames_of_faces(detector, predictor, frames).items():
            v = Video(vtype='mouth')
            v.face = np.array(mouth_frames)
            v.mouth = np.array(mouth_frames)
            v.set_data(mouth_frames)
            yield name, v

    def _mouth_frames_of_faces(self, detector, predictor, frames):
        frameskip = 0
        known_as = dict()
        unknowns = 0
        mouthes = dict()

        for nr, frame in enumerate(frames):
            if nr % self._face_updates == 0:
                known_as.clear()
            if frameskip:
                if frameskip > 10:
                    frameskip = 0
                else:
                    frameskip = max(0, frameskip - 1)
            began_at = time.time()
            if self._preview and not frameskip:
                showframe = frame.copy()
            if self._face_detect_subsample > 1:
                scale = 1.0 / self._face_detect_subsample
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            else:
                scale = 1
            dets = detector(frame, 1)

            for k, d in enumerate(dets):
                shape = predictor(frame, d)

                if nr % self._face_updates == 0:
                    face_encoding = face_recognition.face_encodings(frame, [(shape.rect.left(), shape.rect.top(), shape.rect.right(), shape.rect.bottom())])[0]
                    if self._known_face_encodings:
                        face_distances = face_recognition.face_distance(self._known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] <= 0.6:
                            face_name = self._known_face_names[best_match_index]
                            known_as[k] = face_name
                            self._log.debug('{}: {}: recognized'.format(face_name, nr))
                    if k not in known_as:
                        face_id = 'UNK{}'.format(k)
                        face_name = 'known_{}'.format(len(self._known_face_names))
                        known_as[k] = face_name
                        self._known_face_encodings.append(face_encoding)
                        self._known_face_names.append(face_name)
                        if face_id in mouthes:
                            self._log.debug('{}: {}: promoting from {}'.format(face_name, nr, face_id))
                            mouthes[face_name] = mouthes[face_id]
                            del mouthes[face_id]
                        self._log.debug('{}: {}: ready for recognize'.format(face_name, nr))

                try:
                    face_name = known_as[k]
                except KeyError:
                    face_name = 'UNK{}'.format(unknowns) # ID
                    unknowns = unknowns + 1
                    self._log.debug('{}: {}: detected'.format(face_name, nr))
                try:
                    mouthes[face_name].append(dict(nr=nr, frame=self._mouth_frame_of_face_shaped(shape, frame)))
                except KeyError:
                    mouthes[face_name] = [dict(nr=nr, frame=self._mouth_frame_of_face_shaped(shape, frame))]

                if self._preview and not frameskip:
                    scale = int(1 / scale)
                    cv2.rectangle(showframe, (scale*shape.rect.left(), scale*shape.rect.top()), (scale*shape.rect.right(), scale*shape.rect.bottom()), (255,0,0), 2)
                    cv2.rectangle(showframe, (scale*shape.rect.left(), scale*shape.rect.bottom() - 17), (scale*shape.rect.right(), scale*shape.rect.bottom()), (255, 0, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(showframe, face_name, (scale*shape.rect.left() + 3, scale*shape.rect.bottom() - 3), font, 0.5, (255, 255, 255), 1)

            elapsed = time.time() - began_at
            if self._preview and not frameskip:
                deadline = 1000/self._framerate
                slack = int(deadline - elapsed*1000)
                if slack < 0:
                    frameskip = math.ceil(-slack / deadline)
                cv2.imshow('Video', cv2.cvtColor(showframe, cv2.COLOR_RGB2BGR))
                cv2.waitKey(max(1, slack))

        # Cut off sporagic detection
        self._log.debug('before cutoff: {}'.format({k:[x['nr'] for x in v] for k,v in mouthes.items()}))
        mouthes = {k:v for k,v in mouthes.items() if len(v) > self._face_updates}
        self._log.debug('after cutoff: {}'.format({k:[x['nr'] for x in v] for k,v in mouthes.items()}))

        # XXX implicit reuse of loop counter
        missing = {k:set(range(nr)) - set(x['nr'] for x in mouthes[k]) for k in mouthes}
        for k,v in missing.items():
            for f in sorted(v):
                if f == 0:
                    self._log.debug('{}: {}: interpolating with blank frame'.format(k, f))
                    mouthes[k].append(dict(nr=f, frame=np.zeros((50,100,3), dtype='uint8')))
                else:
                    interp_from = [x for x in mouthes[k] if x['nr'] == (f-1)][0]
                    self._log.debug('{}: {}: interpolating from the previous frame ({})'.format(k, f, interp_from['nr']))
                    mouthes[k].append(dict(nr=f, frame=interp_from['frame']))


        return {k:[x['frame'] for x in sorted(v, key=lambda x: x['nr'])] for k,v in mouthes.items()}

    def _mouth_frame_of_face_shaped(self, shape, frame):
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
    def _cnn_face_detector(*args, **kwargs):
        return (x.rect for x in cnn_face_detector(*args, **kwargs))
