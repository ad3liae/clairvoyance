import asyncio
import random

from lipnet.lipreading.videos import Video
import sys
import os
import functools

from clairvoyance.core import Speaker

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','..','LipNet','common','predictors','shape_predictor_68_face_landmarks.dat')

class FaceRecognitionTask:
    def __init__(self, q):
        self._q = q

    async def do(self, video_path):
        for i in range(1000):
            print("Sending batch #{}".format(i))
            print("Loading data from disk...")
            video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
            if os.path.isfile(video_path):
                video.from_video(video_path)
            else:
                video.from_frames(video_path)
            print("Data loaded.")
            await asyncio.get_event_loop().run_in_executor(None, self._q.put, Speaker(video=video, identity='Speaker #0'))
