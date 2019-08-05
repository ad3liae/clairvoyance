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
                self._log.debug("Loading data from disk...")
                began_at = time.time()
                video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH, preview=self._config.show_frame)
                video.from_array(block, framerate=dec._framerate())
                self._log.debug("Data loaded ({}, {:.02f} sec.).".format(video.data.shape, time.time() - began_at))
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
