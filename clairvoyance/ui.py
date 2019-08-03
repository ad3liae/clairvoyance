def shell():
    import sys
    import logging
    logging.basicConfig(level=logging.DEBUG)

    fn = sys.argv[1]
    Session(fn).invoke()

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
