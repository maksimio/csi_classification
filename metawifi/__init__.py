#TODO Умное чтение вложенных файлов с отсевом не .dat файлов
#TODO MetaWifi - методы с информацией о df (число категорий и т.п.)
from re import I
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

        self.__filter()
        self.__w.hprint(self.__w.INFO, 'MetaWifi: filter df_raw and remove ' + str(self.df_raw.shape[0] - self.df.shape[0]) + ' packets' )

        self.restore_csi()
        self.__w.hprint(self.__w.INFO, 'MetaWifi: get complex CSI dfs in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        self.__num_tones = int(self.df['num_tones'].mean())

        self.__type = 'complex'

        self.__split_csi()
        self.__concat_csi()


    @W.stopwatch
    def __make_df_raw(self) -> None:
        self.df_raw = pd.DataFrame(self.raw)


    @W.stopwatch
    def __filter(self) -> None:
        df = self.df_raw
        self.df = df[(df['csi_len'] != 0)].reset_index(drop=True)


    @W.stopwatch
    def restore_csi(self):
        csi = np.array(self.df['csi'].to_list()) #TODO отлавливать ошибку IndexError: tuple index out of range и выводить ее
        self.df_csi_complex = pd.DataFrame(csi.reshape((csi.shape[0], csi.shape[1] * csi.shape[2]))) #TODO заменить to_list на values
        self.df_csi_abs = self.df_csi_complex.abs()
        self.df_csi_phase = pd.DataFrame(np.angle(self.df_csi_complex.values))

        return self


    def __split_csi(self) -> None:
        if self.__type == 'complex':
            df = self.df_csi_complex
        elif self.__type == 'abs':
            df = self.df_csi_abs
        elif self.__type == 'phase':
            df = self.df_csi_phase

        self.__df_csi_lst = []
        for i in range(int(df.shape[1] / self.__num_tones)):
            item = df[[k + i*self.__num_tones for k in range(0, self.__num_tones)]]
            self.__df_csi_lst.append(item)


    def __concat_csi(self) -> None:
        if self.__type == 'complex':
            self.df_csi_complex = pd.concat(self.__df_csi_lst, axis=1)
        elif self.__type == 'abs':
            self.df_csi_abs = pd.concat(self.__df_csi_lst, axis=1)
        elif self.__type == 'phase':
            self.df_csi_phase = pd.concat(self.__df_csi_lst, axis=1)


    def set_type(self, active_type: str):
        if active_type == 'complex' or active_type == 'abs' or active_type == 'phase':
            self.__type = active_type
            self.__w.hprint(self.__w.SUCCESS, 'MetaWifi: change type to ' + active_type)
            return self
        else:
            self.__w.hprint(self.__w.FAIL, 'MetaWifi: wrong active type in set_type! Exit...')
            exit()


    def get_type(self) -> str:
        return self.__type
        
    
    def smooth(self, window: int=5, win_type: str=None):
        '''https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows'''
        if self.__type == 'complex':
            self.__w.hprint(self.__w.FAIL, 'MetaWifi: in smooth active type can`t be complex! Exit...')
            exit()
        self.__split_csi()

        for i in range(len(self.__df_csi_lst)):
            self.__df_csi_lst[i] = self.__df_csi_lst[i].T.rolling(window, min_periods=1, center=True, win_type=win_type).mean().T

        self.__concat_csi()
        return self