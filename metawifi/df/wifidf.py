from __future__ import annotations
from ..watcher import Watcher as W
from ..log.logcombiner import LogCombiner
import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt


class WifiDf(LogCombiner):
    def __init__(self, dirpath: list = None, categories: list = None) -> None:
        pathes = LogCombiner.subdirs(dirpath)
        LogCombiner.__init__(self, pathes, categories)
        self.__w = W()
        self.combine()

        self.__make_df_raw()
        self.__w.hprint(self.__w.INFO, 'WifiDf: {} to df in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        self.__filter()
        self.__w.hprint(self.__w.INFO, 'WifiDf: filter df_raw and remove ' + str(self.df_raw.shape[0] - self.df.shape[0]) + ' packets')

        self.restore_csi()
        self.__w.hprint(self.__w.INFO, 'WifiDf: get complex CSI dfs in ' + str(round(self.time[-1]['duration'], 2)) + ' seconds')

        self.__num_tones = int(self.df['num_tones'].mean())
        self.__type = 'complex'

        pd.options.mode.chained_assignment = None

    @W.stopwatch
    def __make_df_raw(self) -> None:
        self.df_raw = pd.DataFrame(self.raw)

    @W.stopwatch
    def __filter(self) -> None:
        df = self.df_raw
        self.df = df[(df['csi_len'] != 0) & (df['payload_len'] > 1000)].reset_index(drop=True)

    @W.stopwatch
    def restore_csi(self) -> WifiDf:
        # TODO отлавливать ошибку IndexError: tuple index out of range и выводить ее
        csi = np.array(self.df['csi'].to_list())
        self.df_csi_complex = pd.DataFrame(csi.reshape((csi.shape[0], csi.shape[1] * csi.shape[2])))  # TODO заменить to_list на values
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
            self.__w.hprint(self.__w.SUCCESS,'WifiDf: change type to ' + active_type)
            return self
        else:
            self.__w.hprint(self.__w.FAIL, 'WifiDf: wrong active type in set_type! Exit...')
            exit()


    def get_type(self) -> str:
        return self.__type


    def __relist(f) -> function:
        def wrapper(self, *args, **kwargs) -> WifiDf:
            self.__split_csi()
            f(self, *args, **kwargs)
            self.__concat_csi()
            return self
        return wrapper


    @__relist
    @W.stopwatch
    def smooth(self, width: int = 5, win: str = None):
        '''https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows'''
        if self.__type == 'complex':
            self.__w.hprint(self.__w.FAIL, 'WifiDf: in smooth active type can`t be complex! Exit...')
            exit()

        for i in range(len(self.__df_csi_lst)):
            self.__df_csi_lst[i] = self.__df_csi_lst[i].T.rolling(width, min_periods=1, center=True, win_type=win).mean().T


    @__relist
    @W.stopwatch
    def unjump(self):
        if self.__type != 'phase':
            self.__w.hprint(self.__w.FAIL, 'WifiDf: in unjump active type should be phase! Exit...')
            exit()

        for i in range(len(self.__df_csi_lst)):
            df_diff = self.__df_csi_lst[i].diff(axis=1).fillna(0)
            df_diff[(df_diff < np.pi * 2 - 1) & (df_diff > -np.pi * 2 + 1)] = 0

            for column in self.__df_csi_lst[i]:
                self.__df_csi_lst[i].loc[:, column:][df_diff[column] > 0] -= np.pi * 2
                self.__df_csi_lst[i].loc[:, column:][df_diff[column] < 0] += np.pi * 2


    @__relist
    @W.stopwatch
    def diff(self, order: int = 1):
        for _ in range(order):
            for i in range(len(self.__df_csi_lst)):
                self.__df_csi_lst[i] = self.__df_csi_lst[i].diff(axis=1).fillna(0)


    @__relist
    @W.stopwatch
    def cumsum(self, order: int = 1):
        for _ in range(order):
            for i in range(len(self.__df_csi_lst)):
                self.__df_csi_lst[i] = self.__df_csi_lst[i].cumsum(axis=1)


    @__relist
    @W.stopwatch
    def scale(self, value: int = 1):
        for i in range(len(self.__df_csi_lst)):
            self.__df_csi_lst[i] = self.__df_csi_lst[i] * value


    @__relist
    @W.stopwatch
    def shift(self, to):
        pass

    # Метод проигрывания csi на графике в реальном времени

    def play():
        pass


    def view(self, count: int=5) -> WifiDf:
        if self.__type == 'complex':
            df = self.df_csi_complex
        elif self.__type == 'abs':
            df = self.df_csi_abs
        elif self.__type == 'phase':
            df = self.df_csi_phase

        df.head(count).T.plot()
        plt.show()
        return self


    def prep_csi(self):
        if self.__type == 'complex':
            df = self.df_csi_complex
        elif self.__type == 'abs':
            df = self.df_csi_abs
        elif self.__type == 'phase':
            df = self.df_csi_phase

        df['category'] = self.df['category']
        df['type'] = self.df['type']

        train = df[df['type'] == 'train']
        test = df[df['type'] == 'test']

        x_train = train.drop(['type', 'category'], axis=1)
        y_train = train['category']
        x_test = test.drop(['type', 'category'], axis=1)
        y_test = test['category']

        return x_train, y_train, x_test, y_test

    
    def prep_abs_phase(self):
        df = pd.concat([self.df_csi_abs, self.df_csi_phase], axis=1)

        df['category'] = self.df['category']
        df['type'] = self.df['type']

        train = df[df['type'] == 'train']
        test = df[df['type'] == 'test']

        x_train = train.drop(['type', 'category'], axis=1)
        y_train = train['category']
        x_test = test.drop(['type', 'category'], axis=1)
        y_test = test['category']

        return x_train, y_train, x_test, y_test


    def prep_rssi(self):
        df = pd.DataFrame()
        df[['rssi0', 'rssi1', 'rssi2', 'rssi3']] = self.df[['rssi0', 'rssi1', 'rssi2', 'rssi3']]

        df['category'] = self.df['category']
        df['type'] = self.df['type']

        train = df[df['type'] == 'train']
        test = df[df['type'] == 'test']

        x_train = train.drop(['type', 'category'], axis=1)
        y_train = train['category']
        x_test = test.drop(['type', 'category'], axis=1)
        y_test = test['category']

        return x_train, y_train, x_test, y_test

