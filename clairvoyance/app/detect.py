import asyncio
import itertools
import random
import sys
import os
import functools
import math

import numpy as np
import skvideo.io
from lipnet.lipreading.videos import Video

from clairvoyance.core import Speaker

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','..','LipNet','common','predictors','shape_predictor_68_face_landmarks.dat')

class FaceRecognitionTask:
    def __init__(self, q):
        self._q = q

    async def do(self, video_path):
        dec = VideoDecoder(video_path)
        total = dec.num_blocks()
        for nr,block in dec.decoded_blocks():
            print("Sending batch #{} (of {})".format(nr, total))
            print("Loading data from disk...")
            video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
            video.from_array(block)
            print("Data loaded ({}).".format(video.data.shape))
            await asyncio.get_event_loop().run_in_executor(None, self._q.put, Speaker(video=video, identity='Speaker #0'))

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
