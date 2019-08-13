import asyncio
import random
import functools
import os
import logging
import time
import sys

import numpy as np

class LipReadingTask:
    def __init__(self, config, q):
        self._config = config
        self._q = q
        self._log = logging.getLogger(self.__class__.__name__)
        self._reader = LipReadingTask._reader_from_config(config)

    @staticmethod
    def _reader_from_config(config):
        reader = config.reader

        # XXX insecure
        import importlib
        try:
            mod = importlib.import_module('clairvoyance_{}'.format(reader))
        except ImportError:
            raise ValueError('unknown lip reader: {}'.format(reader))
        try:
            reader_cls = mod.Reader
        except AttributeError as e:
            raise ValueError('invalid lip reader: {} does not define required Reader class'.format(reader))
        try:
            reader_config_cls = reader_cls.Config
        except AttributeError as e:
            raise ValueError('invalid lip reader: {} does not define required Reader.Config class'.format(reader))
        return reader_cls(reader_config_cls())

    async def do(self):
        began_at = time.time()
        self._log.debug('warming up...')
        self._reader.warmup()
        self._log.debug('done ({:.02f} sec).'.format(time.time() - began_at))
        while True:
            speaker = await asyncio.get_event_loop().run_in_executor(None, self._q.get)
            if speaker is not None:
                sys.stdout.write("{}: (detecting)".format(speaker.identity))
                began_at = time.time()

                result = self._reader.do(speaker.video.data)

                sys.stdout.write("\r{}: {} ({:.02f} sec)\n".format(speaker.identity, result, time.time() - began_at))
            else:
                break
