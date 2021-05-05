'''Contains specific functions for plotting'''

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import dtwork.prep as prep


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
col_vals = ['blue', 'orange', 'green', 'red', 'purple',
            'brown', 'pink', 'gray', 'yellow', 'azure']


def csi_plot_types(df, all_in_one=True):
    '''Plots on which csi for each object
    is marked with a different color.'''

    handles = []
    object_types = df['object_type'].unique()
    for obj_type, color in zip(object_types, colors):
        obj_df = df[df['object_type'] == obj_type].drop(columns='object_type').T
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
