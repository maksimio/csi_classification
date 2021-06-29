from .._timerun import stopwatch, HighLight as HL
from .logreader import LogReader
from os.path import join
from os import listdir

class LogCombiner:
    __base_groups = [
        'kitchen', 'room', 'bathroom', 'hall', 'toilet', 'air',
        'bottle', 'thermos', 'grater', 'casserole', 'dish'
    ]
    __hl = HL()
    __dirname_test = 'test'
    __dirname_train = 'train'


    def __init__(self, pathes: list, groups: list=None) -> None:
        self.filelist = []
        self.readers = []
        self.raw = []

        if groups == None:
            self.groups = LogCombiner.__base_groups
            LogCombiner.__hl.hprint(LogCombiner.__hl.WARNING, 'LogCombiner: set base_groups in groups!')
        else:
            self.groups = groups

        self.dir_pathes = pathes
        self.__make_filelist()

    
    @staticmethod
    def train_test(main_path: str):
        if not main_path.endswith('/'):
            main_path += '/'

        train_path = join(main_path, LogCombiner.__dirname_train) + '/'
        test_path = join(main_path, LogCombiner.__dirname_test) + '/'

        return [train_path, test_path]


    def __make_filelist(self):
        for path in self.dir_pathes:
            lst = listdir(path)
            self.filelist += list(map(lambda f: path + f, lst))


    @stopwatch
    def __read(self):
        for fpath in self.filelist:
            self.readers.append(LogReader(fpath).read())
        return self


    @stopwatch
    def _extract(self):
        for logreader in self.readers:
            self.raw += logreader.add()
        return self

    
    def filter():
        pass
    

    def combine(self):
        self.__read()
        LogCombiner.__hl.hprint(LogCombiner.__hl.INFO, 'LogCombiner: read ' + str(len(self.readers)) + ' files in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds (' + str(round(len(self.readers) / self.time[-1]['duration'], 1)) +  ' files/second)')
        self._extract()
        LogCombiner.__hl.hprint(LogCombiner.__hl.INFO, 'LogCombiner: extract files in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        return self