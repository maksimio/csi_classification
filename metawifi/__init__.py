from .watcher import Watcher as W
from .log.logcombiner import LogCombiner
import pandas as pd


class MetaWifi(LogCombiner):
    def __init__(self, dirpath: list=None, categories: list=None) -> None:
        pathes = LogCombiner.subdirs(dirpath)
        LogCombiner.__init__(self, pathes, categories)
        self.__w = W()
        self.combine()
        self.__raw_features = list(self.raw[0].keys())
        self.__make_df_raw()
        self.__w.hprint(self.__w.INFO, 'MetaWifi: [\{\}] to df in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')
    

    @W.stopwatch
    def __make_df_raw(self) -> None:
        self.df_raw = pd.DataFrame(self.raw)