import asyncio
import random
import functools
import os
import logging
import time
import sys

from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
#PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','LipNet','common','dictionaries','grid.txt')
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','LipNet','common','dictionaries','big.txt')
WEIGHT_PATH = os.path.join(CURRENT_PATH,'..','..','LipNet','evaluation','models', 'overlapped-weights368.h5')

class LipReadingTask:
    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 50
    IMAGE_CHANNELS = 3
    FRAMES = 256

    def __init__(self, config, q):
        self._config = config
        self._q = q
        self._log = logging.getLogger(self.__class__.__name__)

    @functools.lru_cache(maxsize=1)
    def decoder(self):
        return Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                       postprocessors=[labels_to_text, Spell(path=PREDICT_DICTIONARY).sentence])

    @functools.lru_cache(maxsize=1)
    def lipnet(self, c, w, h, n, absolute_max_string_len=32, output_size=28):
        lipnet = LipNet(img_c=c, img_w=w, img_h=h, frames_n=n,
                        absolute_max_string_len=absolute_max_string_len, output_size=output_size)

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        lipnet.model.load_weights(WEIGHT_PATH)
        return lipnet

    def _warmup(self):
        self.lipnet(c=self.IMAGE_CHANNELS, w=self.IMAGE_WIDTH, h=self.IMAGE_HEIGHT, n=self.FRAMES)

    async def do(self):
        began_at = time.time()
        self._log.debug('warming up...')
        self._warmup()
        self._log.debug('done ({:.02f} sec).'.format(time.time() - began_at))
        while True:
            speaker = await asyncio.get_event_loop().run_in_executor(None, self._q.get)
            if speaker is not None:
                sys.stdout.write("{}: (detecting)".format(speaker.identity))
                began_at = time.time()
                if K.image_data_format() == 'channels_first':
                    img_c, frames_n, img_w, img_h = speaker.video.data.shape
                else:
                    frames_n, img_w, img_h, img_c = speaker.video.data.shape

                assert (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS) == (img_w, img_h, img_c), '{} != {}'.format((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS), (img_w, img_h, img_c))

                X_data       = np.array([speaker.video.data]).astype(np.float32) / 255
                input_length = np.array([len(speaker.video.data)])

                y_pred         = self.lipnet(c=img_c, w=img_w, h=img_h, n=frames_n).predict(X_data)
                result         = self.decoder().decode(y_pred, input_length)[0]

                sys.stdout.write("\r{}: {} ({:.02f} sec)\n".format(speaker.identity, result, time.time() - began_at))
            else:
                break
