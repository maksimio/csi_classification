'''Contains specific functions for plotting'''

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
col_vals = ['blue', 'orange', 'green', 'red', 'purple',
            'brown', 'pink', 'gray', 'yellow', 'azure']


def csi_plot_types(df, all_in_one=True):
    '''Plots a graph on which csi for
    each object is marked with a different color.'''
    handles = []
    object_types = df['object_type'].unique()
    for obj_type, color in zip(object_types, colors):
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



  # if plot_and_exit:  # TODO check plot code
  #   small_df_train = prep.cut_csi(df_train, 30)  # To make the schedule faster

  #   if False:  # Simple showing:
  #       plot.csi_plot_types(small_df_train)

  #   if True:  # Showing with smoothing and lowering:
  #       df_lst = prep.split_csi(small_df_train)
  #       smoothed_df_lst = prep.smooth(*df_lst)
  #       lowered_df_lst = prep.down(*smoothed_df_lst, level=150)
  #       new_small_df = prep.concat_csi(lowered_df_lst)
  #       plot.csi_plot_types(new_small_df)

  #   if True:  # Wrong showing (smoothing full df):
  #       moothed_df_lst = prep.smooth_savgol(small_df_train)
  #       plot.csi_plot_types(moothed_df_lst)

  #   if True:  # Showing only one path of antennas:
  #       df_lst = prep.split_csi(small_df_train)
  #       plot.csi_plot_types(df_lst[3])

  #   if True:  # Showing smoothed one path and all paths using simple smoothing:
  #       df_lst = prep.split_csi(small_df_train)
  #       smoothed_df_lst = prep.smooth(*df_lst, window=2)
  #       plot.csi_plot_types(smoothed_df_lst[0])
  #       plot.csi_plot_types(prep.concat_csi(smoothed_df_lst))

  #   exit()