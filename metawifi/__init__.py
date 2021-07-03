#TODO Умное чтение вложенных файлов с отсевом не .dat файлов
#TODO MetaWifi - методы с информацией о df (число категорий и т.п.)

from .watcher import Watcher as W
from .log.logcombiner import LogCombiner
import pandas as pd
import numpy as np


class MetaWifi(LogCombiner):
    def __init__(self, dirpath: list=None, categories: list=None) -> None:
        pathes = LogCombiner.subdirs(dirpath)
        LogCombiner.__init__(self, pathes, categories)
        self.__w = W()
        self.combine()

        self.__make_df_raw()
        self.__w.hprint(self.__w.INFO, 'MetaWifi: {} to df in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        self.__make_df_raw_filter()

        self.__make_csi()
        self.__w.hprint(self.__w.INFO, 'MetaWifi: get complex CSI dfs in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')


    @W.stopwatch
    def __make_df_raw(self) -> None:
        self.df_raw = pd.DataFrame(self.raw)


    def __make_df_raw_filter(self) -> None:
        df = self.df_raw
        self.df = df[(df['csi_len'] != 0)].reset_index(drop=True)


    @W.stopwatch
    def __make_csi(self) -> None:
        csi = np.array(self.df['csi'].to_list()) #TODO отлавливать ошибку IndexError: tuple index out of range и выводить ее
        self.df_csi_complex = pd.DataFrame(csi.reshape((csi.shape[0], csi.shape[1] * csi.shape[2]))) #TODO заменить to_list на values
        
        self.df_csi_abs = self.df_csi_complex.abs()
        self.df_csi_phase = pd.DataFrame(np.angle(self.df_csi_complex.values))

