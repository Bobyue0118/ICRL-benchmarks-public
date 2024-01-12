import os
import imageio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# import seaborn as sns
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import numpy as np
import sys

cwd = os.getcwd()#返回当前工作路径(CWD: Current Working Directory),是脚本运行的地方
sys.path.append(cwd.replace('/interface', '/utils'))

def plot_curve(draw_keys, x_dict, y_dict, save_name,
               ylim=(0, 1),
               linewidth=3, xlabel=None, ylabel=None, title=None,
               apply_rainbow=False, apply_scatter=False,
               img_size=(8, 5), axis_size=15, legend_size=15):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig = plt.figure(figsize=img_size)
    ax = fig.add_subplot(1, 1, 1)
    from matplotlib.pyplot import cm
    if apply_rainbow:
        color = cm.rainbow(np.linspace(0, 1, len(draw_keys)))
        for key, c in zip(draw_keys, color):
            if apply_scatter:
                plt.scatter(x_dict[key], y_dict[key], label=key, s=linewidth * 7, c=c)
            else:
                plt.plot(x_dict[key], y_dict[key], label=key, linewidth=linewidth, c=c)
    else:
        for key in draw_keys:
            if apply_scatter:
                plt.scatter(x_dict[key], y_dict[key], label=key, s=linewidth * 7)
            else:
                plt.plot(x_dict[key], y_dict[key], label=key, linewidth=linewidth)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%01.2lf'))
    plt.ylim(ylim[0], ylim[1])
    if legend_size is not None:
        plt.legend(fontsize=legend_size, loc='upper right')
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=axis_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=axis_size)
    if title is not None:
        plt.title(title, fontsize=axis_size)
    if not save_name:
        plt.show()
    else:
        plt.savefig('{0}.png'.format(save_name))
    plt.close()


def plot_shadow_curve(draw_keys,
                      x_dict_mean,
                      y_dict_mean,
                      x_dict_std,
                      y_dict_std,
                      ylim=None,
                      title=None,
                      xlabel=None,
                      ylabel=None,
                      plot_name=None,
                      legend_dict=None,
                      linestyle_dict=None,
                      linewidth=3,
                      img_size=(7, 5),
                      axis_size=15,
                      title_size=15,
                      legend_size=15):
    # sns.set()
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig = plt.figure(figsize=img_size)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for key_idx in range(len(draw_keys)):
        key = draw_keys[key_idx]
        plt.fill_between(x_dict_std[key],
                         y_dict_mean[key] - y_dict_std[key],
                         y_dict_mean[key] + y_dict_std[key],
                         alpha=0.2,
                         # color=colors[key_idx],
                         edgecolor="w",
                         # label=key,
                         )
        plt.plot(x_dict_mean[key],
                 y_dict_mean[key],
                 # color=colors[key_idx],
                 linewidth=linewidth,
                 label=key if legend_dict is None else legend_dict[key],
                 linestyle='-' if linestyle_dict is None else linestyle_dict[key])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if legend_size is not None:
        plt.legend(fontsize=legend_size, loc='upper left')  # upper right, lower left
    if title is not None:
        plt.title(title, fontsize=title_size)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=axis_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=axis_size)
    if not plot_name:
        plt.show()
    else:
        plt.savefig('{0}_shadow.png'.format(plot_name))
    plt.close()


if __name__ == '__main__':
    draw_keys = [0]
    y_dict = [[3e-05, 0.0, 0.0, 0.0, 2e-05, 0.0, 0.00013, 0.01977, 0.0, 2e-05, 0.05765, 0, 0, 0, 0, 2e-05, 0.05765, 0.05765, 0.05765, 0.05765, 0.05765, 0, 0.08235, 0, 0.00163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00332, 0.00233, 0.00332, 0.00332, 0.00332, 0.00039, 0.00332, 0.00332, 0.00163, 0.00332, 0.00678, 0.00678, 0.00678],[7e-05, 0.00163, 0.00039, 0.0, 0.00013, 0.0, 0.05765, 9e-05, 0.0, 3e-05, 0.05765, 2e-05, 0, 0, 0, 0, 0.05765, 0.02825, 0.00969, 0.05765, 0.05765, 0, 2e-05, 0.00233, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.00332, 0.00332, 0.00332, 0.00332, 0.00332, 0.00332, 0.00114, 0.00114, 0.00332, 0.00163, 0.00678, 0.00678, 0.00678]]
    y_dict_mean = [np.mean(y_dict, axis=0)]
    y_dict_std = [np.std(y_dict, axis=0)]
    x_dict_mean = [list(range(50))]
    x_dict_std = [list(range(50))]

    plot_shadow_curve(draw_keys,
                      x_dict_mean,
                      y_dict_mean,
                      x_dict_std,
                      y_dict_std,
                      ylim=None,
                      title=None,
                      xlabel=None,
                      ylabel=None,
                      plot_name=None,
                      legend_dict=None,
                      linestyle_dict=None,
                      linewidth=3,
                      img_size=(7, 5),
                      axis_size=15,
                      title_size=15,
                      legend_size=15)
