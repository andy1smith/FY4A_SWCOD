'''
Functions to help format the plots.

Author: Mengying Li.
'''

import matplotlib as mpl
import numpy as np
import math
# import plotting packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



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


def format_axes(ax,xlims,ylims,xticks,yticks,is_xtl,is_ytl):
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
        ax.set_yticklabels([])





def ghi2d_show(F_ghi_2d, channel, ang, vmax, logscale=True):
    font = 15
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th)
    bins_phi = np.arange(-180, 181, d_phi)

    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    # Create a new figure with defined size
    fig = plt.figure(figsize=(5, 4))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=-0.1, hspace=0.0)

    # Apply logarithmic scaling if needed
    if logscale:
        Z = np.log10(F_ghi_2d.T + 1.0)
    else:
        Z = F_ghi_2d.T

    # Create the subplot
    ax1 = fig.add_subplot(gs1[0, 0])

    # The extent parameter defines the bounding box in data coordinates
    extent = [bins_theta[0], bins_theta[-1], bins_phi[0], bins_phi[-1]]

    # Display the image with the appropriate bin edges
    im = ax1.imshow(Z, cmap='Spectral_r', origin="lower", norm=norm, extent=extent)

    # Set x and y labels
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Phi (degrees)')

    # Set ticks for x-axis based on the bins
    ax1.set_xticks(np.arange(0, 90 + 30, 30))  # Set ticks at each bin edge
    ax1.set_yticks(np.arange(-180, 180 + 30, 30))  # Set ticks at each bin edge

    # Add a color bar
    cbar1 = plt.colorbar(im)
    # Add a color bar with custom ticks
    cbar1.set_ticks(np.arange(0, vmax, 0.05))  # Set ticks from 0 to 2 with a step of 0.5
    if logscale:
        cbar1.set_label('log$_{10}$ of intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                        labelpad=0, fontsize=font, family=fontfml)  # ****
    else:
        cbar1.set_label('Intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                        labelpad=0, fontsize=font, family=fontfml)
    cbar1.ax.tick_params(labelsize=font, labelcolor='black')  # ****
    # Show the plot
    fig_dir = "./figures/angular_distribution_unstable_test/"
    plt.savefig(fig_dir + f'Intensity_{channel}_Z={ang}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return None


import matplotlib.colors as mcolors


def ghi2d_show(F_ghi_2d, channel, ang, vmax, logscale=True):
    font = 15
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th)
    bins_phi = np.arange(-180, 181, d_phi)

    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    # Create a new figure with defined size
    fig = plt.figure(figsize=(5, 4))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=-0.1, hspace=0.0)

    # Apply logarithmic scaling if needed
    if logscale:
        Z = np.log10(F_ghi_2d.T + 1.0)
    else:
        Z = F_ghi_2d.T

    # Create the subplot
    ax1 = fig.add_subplot(gs1[0, 0])

    # The extent parameter defines the bounding box in data coordinates
    extent = [bins_theta[0], bins_theta[-1], bins_phi[0], bins_phi[-1]]

    # Display the image with the appropriate bin edges
    im = ax1.imshow(Z, cmap='Spectral_r', origin="lower", norm=norm, extent=extent)

    # Set x and y labels
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Phi (degrees)')

    # Set ticks for x-axis based on the bins
    ax1.set_xticks(np.arange(0, 90 + 30, 30))  # Set ticks at each bin edge
    ax1.set_yticks(np.arange(-180, 180 + 30, 30))  # Set ticks at each bin edge

    # Add a color bar
    cbar1 = plt.colorbar(im)
    # Add a color bar with custom ticks
    cbar1.set_ticks(np.arange(0, vmax, 0.05))  # Set ticks from 0 to 2 with a step of 0.5
    if logscale:
        cbar1.set_label('log$_{10}$ of intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                        labelpad=0, fontsize=font, family=fontfml)  # ****
    else:
        cbar1.set_label('Intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                        labelpad=0, fontsize=font, family=fontfml)
    cbar1.ax.tick_params(labelsize=font, labelcolor='black')  # ****
    # Show the plot
    fig_dir = "./figures/angular_distribution_unstable_test/"
    plt.savefig(fig_dir + f'Intensity_{channel}_Z={ang}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return None

