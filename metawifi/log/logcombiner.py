from .._timerun import stopwatch, HighLight as HL
from .logreader import LogReader
from os.path import join
from os import listdir
import re

class LogCombiner:
    __base_categoriess = [
        'kitchen', 'room', 'bathroom', 'hall', 'toilet', 'air',
        'bottle', 'thermos', 'grater', 'casserole', 'dish'
    ]
    __hl = HL()
    __dirname_test = 'test'
    __dirname_train = 'train'


    def __init__(self, pathes: list, categories: list=None) -> None:
        self.filelist = []
        self.readers = []
        self.raw = []

        if categories == None:
            self.categories = LogCombiner.__base_categoriess
            LogCombiner.__hl.hprint(LogCombiner.__hl.WARNING, 'LogCombiner: set base_categories in categories!')
        else:
            self.categories = categories

        self.categories.sort(key = len)
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

    
    def __get_category(self, filepath):
        for cat in self.categories:
            research = re.search('.*' + cat + '.*', filepath)
            if research:
                return research.group(0)
        

        return 'unknown'


    @stopwatch
    def __read(self):
        for fpath in self.filelist:
            self.readers.append(LogReader(fpath).read())
        return self


    @stopwatch
    def _extract(self):
        for logreader in self.readers:
            self.raw += logreader.add().add('catogory', self.__get_category(logreader.path))
        return self

    
    def filter():
        pass


    def combine(self):
        self.__read()
        LogCombiner.__hl.hprint(LogCombiner.__hl.INFO, 'LogCombiner: read ' + str(len(self.readers)) + ' files in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds (' + str(round(len(self.readers) / self.time[-1]['duration'], 1)) +  ' files/second)')
        self._extract()
        LogCombiner.__hl.hprint(LogCombiner.__hl.INFO, 'LogCombiner: extract files in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        return self