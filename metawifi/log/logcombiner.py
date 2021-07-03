from ..watcher import Watcher as W
from .logreader import LogReader
from os.path import join
from os import listdir
from re import search

class LogCombiner:
    __base_categories = [
        'kitchen', 'room', 'bathroom', 'hall', 'toilet', 'air',
        'bottle', 'thermos', 'grater', 'casserole', 'dish'
    ]
    __unknown = 'unknown'


    def __init__(self, pathes: list, categories: list=None) -> None:
        self.filelist = []
        self.readers = []
        self.raw = []
        self.__w = W()

        if categories == None:
            self.categories = self.__base_categories
            self.__w.hprint(self.__w.WARNING, 'LogCombiner: set base_categories in categories!')
        else:
            self.categories = categories

        self.types = list(set(path.split('/')[-2] for path in pathes))
        self.categories.sort(key = len)
        self.categories.reverse()
        self.dir_pathes = pathes
        self.__make_filelist()

    
    @staticmethod
    def subdirs(main_path: str):
        if not main_path.endswith('/'):
            main_path += '/'
        
        subdirs = listdir(main_path)
        subdirs = [join(main_path, subdir) + '/' for subdir in subdirs]

        return subdirs


    def __make_filelist(self):
        for path in self.dir_pathes:
            lst = listdir(path)
            self.filelist += list(map(lambda f: path + f, lst))

    
    def __get_category(self, filepath):
        for cat in self.categories:
            if search('.*' + cat + '.*', filepath):
                return cat
        
        return self.__unknown

    def __get_type(self, filepath):
        for type in self.types:
            if search('.*' + type + '.*', filepath):
                return type
        
        return self.__unknown


    @W.stopwatch
    def __read(self):
        for fpath in self.filelist:
            self.readers.append(LogReader(fpath).read())
        return self


    @W.stopwatch
    def _extract(self):
        for logreader in self.readers:
            self.raw += logreader.add().add('category', self.__get_category(logreader.path)).add('type', self.__get_type(logreader.path))
        return self


    def combine(self):
        self.__read()
        self.__w.hprint(self.__w.INFO, 'LogCombiner: read ' + str(len(self.readers)) + ' files in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds (' + str(round(len(self.readers) / self.time[-1]['duration'], 1)) +  ' files/second)')
        self._extract()
        self.__w.hprint(self.__w.INFO, 'LogCombiner: extract files in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        return self