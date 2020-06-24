'''Contains specific functions for plotting'''

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
col_vals = ['blue', 'orange', 'green', 'red', 'purple',
            'brown', 'pink', 'gray', 'yellow', 'azure']


def csi_plot_types(df, all_in_one=True):
    '''Строит график, на котором csi для
    каждого объекта обозначены разным цветом.'''
    handles = []
    object_types = df['object_type'].unique()
    for obj_type, color, col_value in zip(object_types, colors, col_vals):
        obj_df = df[df['object_type'] == obj_type]
        obj_df = obj_df.drop(columns='object_type').T
        plt.plot(obj_df, color=color, linewidth=0.75)
        if not all_in_one:
            plt.legend([obj_type], loc='best')
            plt.grid()
            plt.ylabel('Power, mW')
            plt.xlabel('Subcarrier index')
            plt.figure()
        else:
            handles.append(mpatches.Patch(color=color, label=obj_type))
    
    if all_in_one:
        plt.legend(handles=handles)

    plt.grid()
    plt.ylabel('Power, mW')
    plt.xlabel('Subcarrier index')
    plt.show()