import asyncio
import random

class LipReadingTask:
    def __init__(self, q):
        self._q = q

    async def do(self):
        while True:
            i = await asyncio.get_event_loop().run_in_executor(None, self._q.get)
            if i is None:
                break
            else:
                await asyncio.sleep(0.5 + 0.5*random.random())
                print('done: {}'.format(i))
