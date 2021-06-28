from time import time
from datetime import datetime


def stopwatch(f):
    def wrapper(self, *args, **kwargs):
        start_time = time()
        f(self, *args, **kwargs)
        stop_time = time()
        
        if not hasattr(self, 'time'):
            self.time = []

        self.time.append({
            'name': f.__name__, 
            'duration': stop_time - start_time,
            'calltime': datetime.fromtimestamp(start_time) 
            })
            
        return self
    
    return wrapper