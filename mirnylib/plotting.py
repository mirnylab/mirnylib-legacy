#(c) 2012 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu)
# Anton Goloborodko (golobor@mit.edu)

"""
Some nice plotting utilities from Max
These include:

-scatter3D
-showPolymerRasmol
-removeAxes     -removes axes from the plot
-removeBorder  - removes border from the plot
-niceShow  - nicer "plt.show"
"""
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

import pylab
import numpy
np = numpy
import scipy.stats as st
from mpl_toolkits.mplot3d.axes3d import Axes3D

from . import numutils


def listToColormap(colorList, cmapName=None):
    NSeg = len(colorList) - 1
    colorList = np.array(colorList)
    if colorList.min() < 0:
        raise ValueError("Colors should be 0 to 1, or 0 to 255")
    if colorList.max() > 1.:
        if colorList.max() > 255:
            raise ValueError("Colors should be 0 to 1 or 0 to 255")
        else:
            colorList = colorList / 255.
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmapName, colorList, 256)

fallList = ((255, 255, 255), (255, 255, 204),
     (255, 237, 160), (254, 217, 118),
     (254, 178, 76), (253, 141, 60),
     (252, 78, 42), (227, 26, 28),
     (189, 0, 38), (128, 0, 38), (0, 0, 0))

bluesList = ((255, 255, 255), (180, 204, 225),
             (116, 169, 207), (54, 144, 192),
             (5, 112, 176), (4, 87, 135),
             (3, 65, 100), (2, 40, 66),
             (1, 20, 30), (0, 0, 0))


acidBluesList = ((255, 255, 255), (162, 192, 222),
                 (140, 137, 187), (140, 87, 167),
                 (140, 45, 143), (120, 20, 120),
                 (90, 15, 90), (60, 10, 60),
                 (30, 5, 30), (0, 0, 0))


def registerList(mylist, name):
    mymap = listToColormap(mylist, name)
    mymapR = listToColormap(mylist[::-1], name + "_r")
    matplotlib.cm.register_cmap(name, mymap)
    matplotlib.cm.register_cmap(name + "_r", mymapR)

registerList(fallList, "fall")
registerList(bluesList, "blues")
registerList(acidBluesList, "acidblues")


def cmap_map(function=lambda x: x, cmap=plt.cm.get_cmap("jet"), mapRange=[0, 1]):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.

    Also trims the "range[0]:range[1]" fragment from the colormap - use this to cut the part of the "jet" colormap!
    """
    cdict = cmap._segmentdata

    for key in cdict.keys():
        print cdict[key]
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])

    step_list = sum(step_dict.values(), [])
    array = numpy.array
    step_list = array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: array(cmap(step)[0:3])
    old_LUT = array(map(reduced_cmap, mapRange[0] + step_list * (
        mapRange[1] - mapRange[0])))
    new_LUT = array(map(function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = map(lambda x: x + (x[1],), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def showPolymerRasmol(x, y=None, z=None, color="auto"):
    """
    Shows the polymer using rasmol.
    Can't properly treat continuous chains (they will have linkers of 5 balls between chains)
    Accepts data as x,y,z or as an array of Nx3 or 3xN shape"
    Shows system  by drawing spheres
    draws 4 spheres in between any two points (5 * N spheres total)

    Parameters
    ----------
    x : array
        Nx3 or 3xN array, or N-long array
    y,z : array or None
        if array, it corresponds to y coordinate. If none, x is assumed to have 3 coordinates

    """

    import os
    import tempfile
    #if you want to change positions of the spheres along each segment, change these numbers
    #e.g. [0,.1, .2 ...  .9] will draw 10 spheres, and this will look better
    shifts = [0., 0.2, 0.4, 0.6, 0.8]

    if y is None:
        data = numpy.array(x)
    else:
        data = numpy.array([x, y, z])
    if len(data[0]) != 3:
        data = numpy.transpose(data)
    if len(data[0]) != 3:
        print "wrong data!"
        return

    #determining the 95 percentile distance between particles,
    meandist = numpy.percentile(numpy.sqrt(
        numpy.sum(numpy.diff(data, axis=0) ** 2, axis=1)), 95)
    #rescaling the data, so that bonds are of the order of 1. This is because rasmol spheres are of the fixed diameter.
    data /= meandist

    #writing the rasmol script. Spacefill controls radius of the sphere.
    rascript = tempfile.NamedTemporaryFile()
    rascript.write("""wireframe off
    color temperature
    spacefill 100
    background white
    """)
    rascript.flush()

    #creating the array, linearly chanhing from -225 to 225, to serve as an array of colors
    #(rasmol color space is -250 to 250, but it  still sets blue to the minimum color it found and red to the maximum).
    if color == "auto":
        colors = numpy.array([int(
            (j * 450.) / (len(data))) - 225 for j in xrange(len(data))])
    else:
        colors = color

    #creating spheres along the trajectory
    #for speedup I just create a Nx4 array, where first three columns are coordinates, and fourth is the color
    newData = numpy.zeros((len(data) * len(shifts) - (len(shifts) - 1), 4))
    for i in xrange(len(shifts)):
        #filling in the array like 0,5,10,15; then 1,6,11,16; then 2,7,12,17, etc.
        #this is just very fast
        newData[i:-1:len(shifts), :3] = data[:-1] * shifts[i] + \
            data[1:] * (1 - shifts[i])
        newData[i:-1:len(shifts), 3] = colors[:-1]
    newData[-1, :3] = data[-1]
    newData[-1, 3] = colors[-1]

    towrite = tempfile.NamedTemporaryFile()
    towrite.write("%d\n\n" % (len(newData)))  # number of atoms and a blank line after is a requirement of rasmol

    for i in newData:
        towrite.write("CA\t%lf\t%lf\t%lf\t%d\n" % tuple(i))
    towrite.flush()
    #For windows you might need to change the place where your rasmol file is
    if os.name == "posix":  # if linux
        os.system("rasmol -xyz %s -script %s" % (towrite.name, rascript.name))
    else:  # if windows
        os.system("C:/RasWin/raswin.exe -xyz %s -script %s" % (
            towrite.name, rascript.name))


def scatter3D(x, y, z, color='b'):
    """shows a scatterplot in 3D"""

    fig = plt.figure()
    ax = Axes3D(fig)

    if (type(color) == numpy.ndarray) or (type(color) == list):
        color = numpy.array(color, dtype=float)
        color -= color.min()
        color /= float(color.max() - color.min())
        if len(set(color)) > 20:
            for i in xrange(len(x)):
                ax.scatter(x[i], y[i], z[i],
                           c=plt.cm.get_cmap("jet")(color[i]))
        else:
            colors = set(color)
            for mycolor in colors:
                mask = (color == mycolor)
                ax.scatter(x[mask], y[mask], z[mask],
                           c=plt.cm.get_cmap("jet")(mycolor))

    else:
        ax.scatter(x, y, z, c=color)
    plt.show()


def removeAxes(mode="normal", shift=0, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for loc, spine in ax.spines.iteritems():
        if mode == "normal":
            if loc in ['left', 'bottom']:
                if shift != 0:
                    spine.set_position(('outward',
                                        shift))  # outward by 10 points
            elif loc in ['right', 'top']:
                spine.set_color('none')  # don't draw spine
            else:
                raise ValueError('unknown spine location: %s' % loc)
        else:
            if loc in ['left', 'bottom', 'right', 'top']:
                spine.set_color('none')  # don't draw spine
            else:
                raise ValueError('unknown spine location: %s' % loc)


def removeBorder(ax=None):
    removeAxes("all", 0, ax=ax)
    if ax is None:
        ax = plt.gca()
    for _, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
        line.set_visible(False)
    if ax is None:
        ax = plt.axes()
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def niceShow(mytype=None, subplotAdjust=[0.08, 0.12, 0.95, 0.98]):
    if mytype == "log":
        plt.xscale("log")
        plt.yscale("log")

    legend = plt.legend(loc=0, prop={"size": 15})
    if legend is not None:
        legend.draw_frame(False)
    removeAxes(shift=0)
    plt.gcf().subplots_adjust(*subplotAdjust)
    plt.show()


def mat_img(a, cmap="jet", trunk=False, **kwargs):
    "shows an array using imshow with colorbar"
    a = numpy.array(a, float)
    if trunk != False:
        if trunk == True:
            trunk = 0.01
        sa = numpy.sort(a.ravel())
        a[a > sa[(1 - trunk) * len(sa)]] = sa[(1 - trunk) * len(sa)]
        a[a < sa[trunk * len(sa)]] = sa[trunk * len(sa)]
    #plt.ioff()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.imshow(a, interpolation = 'nearest', cmap = cmap)
    #cbar = fig.colorbar(cax)

    def do_all():
        plt.imshow(a, interpolation='nearest', cmap=cmap, **kwargs)
        plt.colorbar()
        plt.show()
    do_all()


def plot_line(a, b, **kwargs):
    """Plot a line y = a * x + b.

    Parameters
    ----------
    a, b : float
        The slope and intercept of the line.
    """
    xlim = pylab.xlim()
    pylab.plot([xlim[0], xlim[1]],
               [a * xlim[0] + b, a * xlim[1] + b],
               **kwargs)


def linear_regression(x, y, a=None, b=None):
    """Calculate coefficients of a linear regression y = a * x + b.

    Parameters
    ----------
    x, y : array_like of float
    a : float, optional
        If specified then the slope coefficient is fixed and equals a.
    b : float, optional
        If specified then the free term is fixed and equals b.

    Returns
    -------
    out : 4-tuple of float
        The structure is (a, b, r, stderr), where
        a -- slope coefficient,
        b -- free term,
        r -- Peason correlation coefficient,
        stderr -- standard deviation.
    """

    if (a is not None and b is None):
        b = numpy.mean([y[i] - a * x[i] for i in range(len(x))])
    elif (a is not None and b is not None):
        pass
    else:
        a, b = numpy.polyfit(x, y, 1)

    r = numpy.corrcoef(x, y)[0, 1]
    stderr = numpy.std([y[i] - a * x[i] - b for i in range(len(x))])

    return (a, b, r, stderr)


def scatter_trend(x, y, **kwargs):
    """Make a scatter plot with a linear regression.

    Parameters
    ----------
    x, y : array_like of float
    plot_type : bool, optional
        The type of a plot.
    plot_trend : bool, optional
        If True then plot a trendline. True by default.
    skip_trend_points : (int,int), optional
        Number of first and last points to skip when plotting a trend.
    show_slope_sigma : bool, optional
        If True then show the standard error of the slope coefficient.
    show_sigma_prediction : bool, optional
        If True then show the standard error of prediction.
    plot_sigmas : bool, optional
        If True then plot confidence intervals of the linear fit.
        False by default.
    label : str, optional
        Add a label for data series.
    title : str, optional
        The title. Empty by default.
    xlabel, ylabel : str, optional
        The axes labels. Empty by default.
    alpha : float, optional
        Transparency of points. 1.0 by default.
    loc : str, optional
        Location of a legend. 'best' by default.
    alpha_legend : float, optional
        Legend box transparency. 0.7 by default.
    """
    x, y = numpy.asarray(x), numpy.asarray(y)
    mask = np.logical_not(numpy.isnan(x) + numpy.isnan(y) + np.isinf(x) + np.isinf(y))
    x, y = x[mask], y[mask]
    (skip_first, skip_last) = kwargs.pop('skip_trend_points', (0, None))
    if not skip_last is None:
        skip_last = -skip_last
    a, b, r, p, slope_sigma = st.linregress(
        x[skip_first:skip_last],
        y[skip_first:skip_last])
    stderr = numpy.std(y - a * x - b)

    if kwargs.pop('show_slope_sigma', True):
        equation_label = '$y\,=\,({:.3f}\pm{:.3f})x\,+\,{:.3f}$'.format(
            a, 2 * slope_sigma, b)
    else:
        equation_label = '$y\,=\,{:.3f}x\,+\,{:.3f}$'.format(a, b)
    r2_label = '$R^2=\,{:.3f}$'.format(r * r)
    sigma_label = '$\sigma\,=\,{:.3f}$'.format(stderr)
    label = ('{}, {}').format(equation_label, r2_label)
    if 'label' in kwargs:
        label = kwargs['label'] + '\n' + label
    if kwargs.pop('show_sigma_prediction', True):
        label = label + '\n' + sigma_label

    if skip_first or skip_last:
        label += '\n$omit\,{}\,terminal\,points$'.format(
            (skip_first,
             - skip_last if skip_last else 0))

    plot_type = kwargs.get('plot_type', 'scatter')
    color = kwargs.get('c', 'b')
    if plot_type == 'scatter':
        plt.scatter(x[skip_first:skip_last],
                    y[skip_first:skip_last],
                    edgecolor=color,
                    facecolor=color,
                    alpha=kwargs.get('alpha', 1.0))
    elif plot_type == 'line':
        plt.plot(x[skip_first:skip_last],
                 y[skip_first:skip_last],
                 color=color,
                 alpha=kwargs.get('alpha', 1.0),
                 marker='.',
                 markersize=10.0)

    if skip_first:
        plt.scatter(x[0:skip_first],
                    y[0:skip_first],
                    edgecolor=color,
                    facecolor='w',
                    alpha=kwargs.get('alpha', 1.0))
    if skip_last:
        plt.scatter(x[skip_last:],
                    y[skip_last:],
                    edgecolor=color,
                    facecolor='w',
                    alpha=kwargs.get('alpha', 1.0))

    plt.title(kwargs.get('title', ''))
    plt.xlabel(kwargs.get('xlabel', ''))
    plt.ylabel(kwargs.get('ylabel', ''))

    if kwargs.get('plot_trend', True):
        plt.plot([min(x), max(x)],
                 [a * min(x) + b, a * max(x) + b],
                 color=color,
                 linestyle='--',
                 label=label)
    if kwargs.pop('plot_sigmas', False):
        for i in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
            plt.plot([min(x), max(x)],
                     [a * min(x) + b + i * stderr,
                      a * max(x) + b + i * stderr],
                     'r--')
    legend = plt.legend(loc=kwargs.get('loc', 'best'))
    legend_frame = legend.get_frame()
    legend_frame.set_alpha(kwargs.get('alpha_legend', 0.7))

def plot_matrix_3d(matrix, **kwargs):
    import mpl_toolkits.mplot3d.axes3d as pylab3d
    ax = pylab3d.Axes3D(pylab.gcf())
    x = kwargs.get('x', numpy.arange(matrix.shape[1]))
    y = kwargs.get('y', numpy.arange(matrix.shape[0]))
    matrix = np.copy(matrix)
    matrix[np.isnan(matrix)] = np.nanmin(matrix)
    X, Y = numpy.meshgrid(x, y)

    plot_type = kwargs.get('plot_type', 'surface')
    if plot_type == 'surface':
        ax.plot_surface(X, Y, matrix, rstride=1, cstride=1,
                        cmap=pylab.cm.get_cmap("jet"), antialiased=False)
    elif plot_type == 'wireframe':
        ax.plot_wireframe(X, Y, matrix, cmap=pylab.cm.get_cmap("jet"))
    elif plot_type == 'scatter':
        ax.scatter3D(numpy.ravel(X), numpy.ravel(Y), numpy.ravel(matrix))
    elif plot_type == 'contour':
        num_contours = kwargs.get('num_contours', 50)
        ax.contour3D(X, Y, matrix, num_contours, cmap=pylab.cm.get_cmap("jet"))
    elif plot_type == 'contourf':
        num_contours = kwargs.get('num_contours', 50)
        ax.contourf3D(X, Y, matrix, num_contours,
                      cmap=pylab.cm.get_cmap("jet"))
    else:
        raise StandardError('Unknown plot type: %s' % (plot_type,))

    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    ax.set_zlabel(kwargs.get('zlabel', ''))
    ax.set_title(kwargs.get('title', ''))


def plot_matrix(matrix, **kwargs):
    """Plot a 2D array with a colorbar.

    Parameters
    ----------

    matrix : a 2d numpy array
        A 2d array to plot
    clip_min : float, optional
        The lower clipping value. If an element of a matrix is <clip_min, it is
        plotted as clip_min.
    clip_max : float, optional
        The upper clipping value.
    label : str, optional
        Colorbar label
    ticklabels1 : list, optional
        Custom tick labels for the first dimension of the matrix.
    ticklabels2 : list, optional
        Custom tick labels for the second dimension of the matrix.
    """
    clip_min = kwargs.pop('clip_min', -numpy.inf)
    clip_max = kwargs.pop('clip_max', numpy.inf)

    if 'ticklabels1' in kwargs:
        plt.yticks(range(matrix.shape[0]))
        plt.gca().set_yticklabels(kwargs.pop('ticklabels1'))

    if 'ticklabels2' in kwargs:
        plt.xticks(range(matrix.shape[1]))
        plt.gca().set_xticklabels(kwargs.pop('ticklabels2'))

    plt.imshow(
        numpy.clip(matrix, a_min=clip_min, a_max=clip_max),
        interpolation='nearest',
        **kwargs)
    if 'label' not in kwargs:
        plt.colorbar()
    else:
        plt.colorbar().set_label(kwargs['label'])



def plot_function(function, **kwargs):
    """Plot the values of a 1-D function on a lattice
    The values of the argument may be supplied as an array or as the range+step
    combination.

    Parameters
    ----------
    function : function
        A 1-D function to plot
    x : array_like of float, optional
        The values of the argument to plot
    x_range : (float, float, float), optional
        The range of the argument and the step in the format (x_min, x_max, step)
    plot_type : {'line', 'scatter'}
        The type of plot, a continuous line or a scatter plot. 'line' by default.
    """

    if 'x' in kwargs and 'x_range' in kwargs:
        raise Exception('Please supply either x or x_range, but not both')

    if 'x' in kwargs:
        x = kwargs.pop('x')
    elif 'x_range' in kwargs:
        x_range = kwargs.pop('x_range')
        x = numpy.arange(x_range[0], x_range[1], x_range[2])

    y = numpy.array([function(i) for i in x], dtype=float)

    plot_type = kwargs.pop('plot_type', 'line')
    if plot_type == 'line':
        plt.plot(x, y, **kwargs)
    elif plot_type == 'scatter':
        plt.scatter(x, y, **kwargs)
    else:
        raise Exception('An unknown type of plot: {0}'.format(plot_type))


def plot_loglog_density(x, bins=10, **kwargs):
    bins = numpy.asarray(numutils.logbins(1, max(x), N_in=bins))
    binsizes = bins[1:] - bins[:-1]
    binmids = bins[:-1] + binsizes / 2
    # Calculate density so that it integrates to one.
    avg_counts = numpy.histogram(x, bins=bins, density=True)[0]
    binmids = binmids[avg_counts != 0]
    avg_counts = avg_counts[avg_counts != 0]
    scatter_trend(numpy.log10(binmids), numpy.log10(avg_counts), **kwargs)


def plot_function_3d(x, y, function, **kwargs):
    """Plot values of a function of two variables in 3D.

    Parameters
    ----------
    x, y : array_like of float
        The values of the arguments to plot.
    function : function
        The function to plot.
    plot_type : {'surface', 'wireframe', 'scatter', 'contour', 'contourf'}
        The type of a plot, see
        `scipy cookbook <http://www.scipy.org/Cookbook/Matplotlib/mplot3D>`_
        for examples. The default value is 'surface'.
    num_contours : int
        The number of contours to plot, 50 by default.
    xlabel, ylabel, zlabel : str, optional
        The axes labels. Empty by default.
    title : str, optional
        The title. Empty by default.

    See also
    --------
    More on 3D plotting in pylab:
    http://www.scipy.org/Cookbook/Matplotlib/mplot3D

    """
    Z = []
    for y_value in y:
        Z.append([])
        for x_value in x:
            Z[-1].append(function(x_value, y_value))
    Z = numpy.array(Z)
    plot_matrix_3d(Z, x=x, y=y, **kwargs)


def plot_function_contour(x, y, function, **kwargs):
    """Make a contour plot of a function of two variables.

    Parameters
    ----------
    x, y : array_like of float
        The positions of the nodes of a plotting grid.
    function : function
        The function to plot.
    filling : bool
        Fill contours if True (default).
    num_contours : int
        The number of contours to plot, 50 by default.
    xlabel, ylabel : str, optional
        The axes labels. Empty by default.
    title : str, optional
        The title. Empty by default.

    """
    X, Y = numpy.meshgrid(x, y)
    Z = []
    for y_value in y:
        Z.append([])
        for x_value in x:
            Z[-1].append(function(x_value, y_value))
    Z = numpy.array(Z)
    num_contours = kwargs.get('num_contours', 50)
    if kwargs.get('filling', True):
        pylab.contourf(X, Y, Z, num_contours, cmap=pylab.cm.get_cmap("jet"))
    else:
        pylab.contour(X, Y, Z, num_contours, cmap=pylab.cm.get_cmap("jet"))
    pylab.xlabel(kwargs.get('xlabel', ''))
    pylab.ylabel(kwargs.get('ylabel', ''))
    pylab.title(kwargs.get('title', ''))


def average_3d_data(x, y, z, nbins):
    """Breaks the xy plane into square regions and calculates an average for
    every region.
    """

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    if (x_max - x_min) > (y_max - y_min):
        delta = (x_max - x_min) / nbins
        nbins_x = nbins + 1
        nbins_y = int(numpy.ceil((y_max - y_min) / delta))
    else:
        delta = (y_max - y_min) / nbins
        nbins_y = nbins
        nbins_x = int(numpy.ceil((x_max - x_min) / delta))
    nbins_x += 1
    nbins_y += 1
    x_min -= delta / 2.0
    y_min -= delta / 2.0
    x_max += delta / 2.0
    y_max += delta / 2.0

    if not issubclass(numpy.ndarray, type(x)):
        x = numpy.array(x)
    if not issubclass(numpy.ndarray, type(y)):
        y = numpy.array(y)
    if not issubclass(numpy.ndarray, type(z)):
        z = numpy.array(z)

    matrix = numpy.zeros(shape=(nbins_x, nbins_y), dtype=float)
    for i in range(nbins_x):
        for j in range(nbins_y):
            lower_x = x_min + i * delta
            upper_x = x_min + (i + 1) * delta
            lower_y = y_min + j * delta
            upper_y = y_min + (j + 1) * delta
            mask = ((x >= lower_x) * (x < upper_x) * (
                y >= lower_y) * (y < upper_y))
            if numpy.any(mask):
                matrix[i, j] = numpy.mean(z[mask])
            else:
                matrix[i, j] = numpy.nan
    return matrix


def plot_average_3d(x, y, z, nbins, **kwargs):
    """Breaks the xy plane into square regions and plots an average for every
    region.
    """

    matrix = average_3d_data(x, y, z, nbins)
    plot_matrix_3d(matrix, **kwargs)


def histogram2d(x, y, bins=10):
    hist, xedges, yedges = numpy.histogram2d(x, y, bins=bins)
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    plot_matrix(hist, extent=extent, aspect='auto')


def bar_chart(y, labels=None, yerr=None, **kwargs):
    """
    Show a categorical bar chart

    This function is based on the code from
    http://www.scipy.org/Cookbook/Matplotlib/BarCharts
    """
    if hasattr(y, 'keys') and hasattr(y, 'values'):
        items = list(y.iteritems())
        items.sort(key=lambda x: x[0])
        labels = [i[0] for i in items]
        y = [i[1] for i in items]
    assert len(y) == len(
        labels), 'The lengths of dataset and labels do not match'

    rotate_labels = kwargs.get('rotate_labels', True)

    width = kwargs.pop('width', 0.4)
    xlocs = numpy.array(range(len(y))) + 0.5
    ecolor = kwargs.pop('ecolor', 'k')
    elinewidth = kwargs.pop('elinewidth', 1.0)
    plt.bar(xlocs, y, yerr=yerr, width=width, ecolor=ecolor,
            error_kw={'elinewidth': elinewidth},
            **kwargs)
    plt.xticks(xlocs + width / 2, labels)
    plt.xlim(0, xlocs[-1] + width * 2)
    plt.ylim(0, max(y) + (max(y) - min(y)) * 0.10)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    if rotate_labels:
        plt.gcf().autofmt_xdate()
