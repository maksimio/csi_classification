'''Содержит специфические функции для постройки графиков'''

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
col_vals = ['blue', 'orange', 'green', 'red', 'purple',
            'brown', 'pink', 'gray', 'yellow', 'azure']


def show():
    plt.show()


def csi_plot_ez(df, do_show=True):
    '''Строит график csi без
    дополнительных преобразований.'''
    df = df.drop(['object_type'], axis=1).T
    plt.plot(df, color="#9467bd")
    if do_show:
        plt.show()
    else:
        plt.figure()


def csi_plot_types(df, all_in_one=True):
    '''Строит график, на котором csi для
    каждого объекта обозначены разным цветом.'''
    object_types = df['object_type'].unique()
    for obj_type, color, col_value in zip(object_types, colors, col_vals):
        obj_df = df[df['object_type'] == obj_type]
        obj_df = obj_df.drop(columns='object_type').T
        plt.plot(obj_df, color=color, linewidth=0.87)
        if not all_in_one:
            plt.legend([obj_type], loc='best')
            plt.grid()
            plt.ylabel('Power, mW')
            plt.xlabel('Subcarrier index')
            plt.figure()
        else:
            print('Color =', col_value, '\tobj_type =', obj_type)

    # TODO  нормальная легенда
    plt.grid()
    plt.ylabel('Power, dbm')
    plt.xlabel('Subcarriers')
    plt.show()


def csi_plot(df_lst):

    for i in range(len(df_lst)):
        df_lst[i] = df_lst[i][(df_lst[i]['object_type'] == 'air')]
        df_lst[i] = df_lst[i].drop(columns='object_type').T

    p0 = plt.plot(df_lst[0], color="#9467bd", linewidth=0.85)
    p1 = plt.plot(df_lst[1], color='#2ca02c', linewidth=0.85)
    p2 = plt.plot(df_lst[2], color='#d62728', linewidth=0.85)
    # p3 = plt.plot(df_lst[3],color='#d62728', linewidth=0.85)

    plt.grid()
    plt.ylabel('Power, mW')
    plt.xlabel('Subcarrier index')

    l1 = mpatches.Patch(color='#9467bd', label='01:20')
    l2 = mpatches.Patch(color='#2ca02c', label='07:23')
    l3 = mpatches.Patch(color='#d62728', label='13:25')
    # l4 = mpatches.Patch(color='#d62728', label='h_22')
#    plt.legend(handles=[l1, l2, l3, l4])
    plt.legend(handles=[l1, l2, l3])
    plt.show()
