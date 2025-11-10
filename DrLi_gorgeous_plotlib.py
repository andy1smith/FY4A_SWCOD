'''
Functions to help format the plots.

Author: Mengying Li.
'''

import matplotlib as mpl
import numpy as np
from matplotlib.ticker import AutoMinorLocator,MultipleLocator,LogLocator

__all__ = [
            "make_cmap",
            "lambda_axis",
            "format_axes",
        ]

# define costom color map
def make_cmap(colors, position=None, bit=False):
    """
    Generate customized color map based on input colors. 
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    Position contains values from 0 to 1 to dictate the location of each color.

    Parameters
    ----------
    colors : list of tuples
            The tuples which contain RGB values. The RGB values may either be in 8-bit [0 to 255] 
            (in which bit must be set to True when called) or arithmetic [0 to 1] (default).

    Returns
    -------
    cmap : a color map with equally spaced colors.
    """
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


def lambda_axis(ax,new_tick_locations,font,fontfml,is_xlab):
    """
    Add secondary x-axis for wavelength [um].

    Parameters
    ----------
    ax : the axes of the primary x-axis for wavenumber [cm-1].
    new_tick_locations: array_like, the tick locations for wavelength axes.
    font,fontfml: font size and font family for the ticks and tick labels.
    is_xlab: bool, indicator of whether to add axis labels.

    Returns
    -------
    None.
    """
    def tick_function(X):
        V = 1e4/X
        return ["%.1f" % z for z in V]
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(new_tick_locations)
    if (is_xlab):
        ax3.set_xticklabels(tick_function(new_tick_locations),fontsize=font,family=fontfml)
    else:
        ax3.set_xticklabels([])
    return ax3


def format_axes(ax,xlims,ylims,xticks,yticks,is_xtl,is_ytl,ylabel=[]):
    """
    Format given axes.

    Parameters
    ----------
    ax : the axes to be formatted. 
    xlims,ylims: [min, max] list, x and y axis limits.
    xticks,yticks: list, x and y axis ticks.
    is_xtl,is_ytl: bool, indicator of whether to add axis labels.

    Returns
    -------
    None.
    """
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if (not is_xtl): 
        ax.set_xticklabels([])
    if (not is_ytl):
        ax.set_yticklabels(ylabel)