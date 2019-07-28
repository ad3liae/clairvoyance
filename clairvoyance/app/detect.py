import asyncio
import random

class FaceRecognitionTask:
    def __init__(self, q):
        self._q = q

    async def do(self):
        for i in range(1000):
            await asyncio.sleep(random.random())
            print('detected: {}'.format(i))
            await asyncio.get_event_loop().run_in_executor(None, self._q.put, i)
