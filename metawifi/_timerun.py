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


class HighLight:
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    INFO = '\033[94m'
    BOLD = '\033[1m'
    __ENDC = '\033[0m'

    __calls = []
    __start_time = None


    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HighLight, cls).__new__(cls)
        return cls.instance


    def hprint(self, m_type, msg):
        t = datetime.now().strftime('%H:%M:%S --> ')
        if type(m_type) == list:
            s = t + ''.join(m_type) + msg + HighLight.__ENDC
        elif type(m_type) == str:
            s = t + m_type + msg + HighLight.__ENDC
        else:
           raise ValueError('M_type should be list or str!') 
        print(s)