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
import matplotlib.pyplot as plt 

import pylab
import numpy 

def cmap_map(function,cmap,mapRange = [0,1]):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    
    Also trims the "range[0]:range[1]" fragment from the colormap - use this to cut the part of the "jet" colormap! 
    """
    cdict = cmap._segmentdata
    
    for key in cdict.keys():
        print cdict[key]
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[0], cdict[key])
    
    step_list = sum(step_dict.values(), [])
    array = numpy.array
    step_list = array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : array(cmap(step)[0:3])
    old_LUT = array(map( reduced_cmap, mapRange[0] + step_list * (mapRange[1] - mapRange[0]) ))
    new_LUT = array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def showPolymerRasmol(x,y=None,z=None):
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
    
    import os, tempfile      
    #if you want to change positions of the spheres along each segment, change these numbers
    #e.g. [0,.1, .2 ...  .9] will draw 10 spheres, and this will look better
    shifts = [0.,0.2,0.4,0.6,0.8]
    
    if y == None: data = numpy.array(x)
    else: data = numpy.array([x,y,z])
    if len(data[0]) != 3: 
        data = numpy.transpose(data)
    if len(data[0]) != 3:
        print "wrong data!"
        return
    
    #determining the 95 percentile distance between particles,  
    meandist = numpy.percentile(numpy.sqrt(numpy.sum(numpy.diff(data,axis = 0)**2,axis = 1)),95)
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
    colors = numpy.array([int((j*450.)/(len(data)))-225 for j in xrange(len(data))])    
    
    #creating spheres along the trajectory
    #for speedup I just create a Nx4 array, where first three columns are coordinates, and fourth is the color      
    newData = numpy.zeros((len(data) * len(shifts) - (len(shifts) - 1) ,4))  
    for i in xrange(len(shifts)):            
        #filling in the array like 0,5,10,15; then 1,6,11,16; then 2,7,12,17, etc. 
        #this is just very fast
        newData[i:-1:len(shifts),:3] = data[:-1] * shifts[i] + data[1:] * ( 1 - shifts[i])            
        newData[i:-1:len(shifts),3] = colors[:-1]
    newData[-1,:3] = data[-1]
    newData[-1,3] = colors[-1]
                
    towrite = tempfile.NamedTemporaryFile()
    towrite.write("%d\n\n"%(len(newData)))  #number of atoms and a blank line after is a requirement of rasmol
        
    for i in newData:                     
        towrite.write("CA\t%lf\t%lf\t%lf\t%d\n" % tuple(i)) 
    towrite.flush()
    #For windows you might need to change the place where your rasmol file is  
    if os.name == "posix":  #if linux 
        os.system("rasmol -xyz %s -script %s" % (towrite.name, rascript.name))
    else:     #if windows 
        os.system("C:/RasWin/raswin.exe -xyz %s -script %s" % (towrite.name, rascript.name))

def scatter3D(x, y, z, color):
    """shows a scatterplot in 3D"""
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if (type(color) == numpy.ndarray) or (type(color) == list):
        color = numpy.array(color,dtype = float)
        color -= color.min()
        color /= float(color.max() - color.min() )
        if len(set(color)) > 20:                        
            for i in xrange(len(x)):
                ax.scatter(x[i], y[i], z[i], c=plt.cm.get_cmap("jet")(color[i]))
        else:
            colors = set(color)
            for mycolor in colors:
                mask = (color == mycolor)
                ax.scatter(x[mask],y[mask],z[mask],c=plt.cm.get_cmap("jet")(mycolor))

    else: ax.scatter(x,y,z,c=color)
    plt.show()
                
def removeAxes(mode = "normal",shift = 0, ax = None):
    if ax == None: 
        ax = plt.gca()
    for loc, spine in ax.spines.iteritems():
        if mode == "normal":
            if loc in ['left','bottom']:
                if shift != 0: spine.set_position(('outward',shift)) # outward by 10 points
            elif loc in ['right','top']:
                spine.set_color('none') # don't draw spine
            else:
                raise ValueError('unknown spine location: %s'%loc)
        else:
            if loc in ['left','bottom','right','top']:
                spine.set_color('none') # don't draw spine
            else:
                raise ValueError('unknown spine location: %s'%loc)
            
def removeBorder(ax = None):
    removeAxes("all",0,ax = ax)
    if ax == None: ax = plt.gca()    
    for _, line in enumerate(ax.get_xticklines() + ax.get_yticklines()): 
        line.set_visible(False)     
    if ax == None: ax = plt.axes()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
def niceShow(mytype = None):
    if mytype == "log":
        plt.xscale("log")
        plt.yscale("log")

    legend = plt.legend(loc=0,prop={"size":15})
    if legend != None:
        legend.draw_frame(False)
    removeAxes(shift = 0)
    plt.gcf().subplots_adjust(left=0.07, bottom=0.12, top=0.98, right=0.98)
    plt.show()
        
def mat_img(a,cmap="hot_r",trunk = False, **kwargs):
    "shows an array using imshow with colorbar"
    a = numpy.array(a,float)
    if trunk != False:
        if trunk == True:
            trunk = 0.01
        sa = numpy.sort(a.ravel())
        a[a>sa[(1 - trunk) * len(sa)]] = sa[(1 - trunk) * len(sa)]
        a[a<sa[trunk * len(sa)]] = sa[trunk * len(sa)]
    #plt.ioff() 
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.imshow(a, interpolation = 'nearest', cmap = cmap)
    #cbar = fig.colorbar(cax)
    
    def do_all():
        plt.imshow(a, interpolation = 'nearest', cmap = cmap,**kwargs)
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
    ylim = pylab.ylim()
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

    if (a!=None and b==None):
        b = numpy.mean([y[i] - a * x[i] for i in range(len(x))])
    elif (a!=None and b!= None):
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
    plot_trend : bool, optional
        If True then plot a trendline. True by default.
    plot_sigmas : bool, optional
        If True then plot confidence intervals of the linear fit.
        False by default.
    title : str, optional
        The title. Empty by default.
    xlabel, ylabel : str, optional
        The axes labels. Empty by default.
    alpha : float, optional
        Transparency of points. 1.0 by default
    alpha_legend : float, optional
        Legend box transparency. 1.0 by default
    """
    a, b, r, stderr = linear_regression(x, y)
    pylab.title(kwargs.get('title', ''))
    pylab.xlabel(kwargs.get('xlabel', ''))
    pylab.ylabel(kwargs.get('ylabel', ''))
    scat_plot = pylab.scatter(x, y,
                              c=kwargs.get('c', 'b'),
                              alpha=kwargs.get('alpha', 1.0))
    scat_plot.set_label(
        '$y\,=\,%.3fx\,+\,%.3f$, $R^2=\,%.3f$ \n$\sigma\,=\,%.3f$' % (
            a, b, r*r, stderr))
    legend = pylab.legend(loc='upper left')
    legend_frame = legend.get_frame()
    legend_frame.set_alpha(kwargs.get('alpha_legend', 1.0))
    if kwargs.get('plot_trend', True):
        pylab.plot([min(x), max(x)],
                   [a*min(x)+b, a*max(x)+b])
    if kwargs.get('plot_sigmas', False):
        for i in [-3.0,-2.0,-1.0,1.0,2.0,3.0]:
            pylab.plot([min(x), max(x)],
                       [a*min(x)+b+i*stderr, a*max(x)+b+i*stderr],
                       'r--')

def plot_function_3d(x, y, function, **kwargs):
    """Plot values of a function of two variables in 3D.

    Parameters
    ----------
    x, y : array_like of float
        The plotting range.
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
    import mpl_toolkits.mplot3d.axes3d as pylab3d
    ax = pylab3d.Axes3D(pylab.gcf())
    X, Y = numpy.meshgrid(x, y)
    Z = []
    for y_value in y:
        Z.append([])
        for x_value in x:
            Z[-1].append(function(x_value, y_value))
    Z = numpy.array(Z)
    plot_type = kwargs.get('plot_type', 'surface')
    if plot_type == 'surface':
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=pylab.cm.jet)
    elif plot_type == 'wireframe':
        ax.plot_wireframe(X, Y, Z, cmap=pylab.cm.jet)
    elif plot_type == 'scatter':
        ax.scatter3D(numpy.ravel(X), numpy.ravel(Y), numpy.ravel(Z))
    elif plot_type == 'contour':
        num_contours = kwargs.get('num_contours', 50)
        ax.contour3D(X, Y, Z, num_contours, cmap=pylab.cm.jet)
    elif plot_type == 'contourf':
        num_contours = kwargs.get('num_contours', 50)
        ax.contourf3D(X, Y, Z, num_contours, cmap=pylab.cm.jet)
    else:
        raise PyteomicsError('Unknown plot type: %s' % (plot_type,))
    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    ax.set_zlabel(kwargs.get('zlabel', ''))
    ax.set_title(kwargs.get('title', ''))

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
        pylab.contourf(X, Y, Z, num_contours, cmap=pylab.cm.jet)
    else:
        pylab.contour(X, Y, Z, num_contours, cmap=pylab.cm.jet)
    pylab.xlabel(kwargs.get('xlabel', ''))
    pylab.ylabel(kwargs.get('ylabel', ''))
    pylab.title(kwargs.get('title', ''))