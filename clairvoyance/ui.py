def shell():
    import sys
    import logging
    import getopt
    from clairvoyance.core import Config

    config = Config()
    opts, fns = getopt.gnu_getopt(sys.argv[1:], '', ['help', 'debug', 'show-frame', 'framerate=', 'face-updates=', 'face-detector='])
    for o,a in opts:
        if o in ['--help']:
            config.help = True
        if o in ['--debug']:
            config.debug = True
        if o in ['--show-frame']:
            config.show_frame = True
        if o in ['--framerate']:
            config.framerate = float(a)
        if o in ['--face-updates']:
            config.face_updates = int(a)
        if o in ['--face-detector']:
            config.face_detector = a

    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)

    if not config.help and fns:
        Session(fns[0]).invoke()
    else:
        print(help(), file=sys.stderr)

def help():
    import sys

    def opt(o, d):
        return r'  {{:{0}s}}{{}}'.format(12).format(o, d)

    return '\n'.join([
        version(),
        '',
        'usage: {} [options] <stream>'.format(sys.argv[0]),
        '',
        'OPTIONS',
        '',
        opt('--help', 'Shows this message'),
        opt('--debug', 'Shows debug message'),
    ])

def version():
    import pkg_resources
    return '\n'.join([
        'Clairvoyance {}: concurrent lip-reader for the smart masses'.format(pkg_resources.get_distribution('clairvoyance').version),
        'Copyright (C) 2019 Ken-ya and Takahiro Yoshimura.  All rights reserved.',
        'Licensed under the terms of GNU General Public License Version 3 or later.',
    ])

class Session:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self):
        import multiprocessing as mp
        q = mp.Queue()
        p1 = mp.Process(target=Session._invoke_detector, args=(q, self._fn, ))
        p2 = mp.Process(target=Session._invoke_reader, args=(q, ))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    @staticmethod
    def _invoke_detector(q, fn):
        from clairvoyance.app.detect import FaceRecognitionTask
        Session._run_async(FaceRecognitionTask(q).do(fn))

    @staticmethod
    def _invoke_reader(q):
        from clairvoyance.app.read import LipReadingTask
        Session._run_async(LipReadingTask(q).do())

    @staticmethod
    def _run_async(coro):
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
