import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

class AnnotatedCursor(Cursor):
    """
    A crosshair cursor like `~matplotlib.widgets.Cursor` with a text showing \
    the current coordinates.

    For the cursor to remain responsive you must keep a reference to it.
    The data of the axis specified as *dataaxis* must be in ascending
    order. Otherwise, the `numpy.searchsorted` call might fail and the text
    disappears. You can satisfy the requirement by sorting the data you plot.
    Usually the data is already sorted (if it was created e.g. using
    `numpy.linspace`), but e.g. scatter plots might cause this problem.
    The cursor sticks to the plotted line.

    Parameters
    ----------
    line : `matplotlib.lines.Line2D`
        The plot line from which the data coordinates are displayed.

    numberformat : `python format string <https://docs.python.org/3/\
    library/string.html#formatstrings>`_, optional, default: "{0:.4g};{1:.4g}"
        The displayed text is created by calling *format()* on this string
        with the two coordinates.

    offset : (float, float) default: (5, 5)
        The offset in display (pixel) coordinates of the text position
        relative to the cross hair.

    dataaxis : {"x", "y"}, optional, default: "x"
        If "x" is specified, the vertical cursor line sticks to the mouse
        pointer. The horizontal cursor line sticks to *line*
        at that x value. The text shows the data coordinates of *line*
        at the pointed x value. If you specify "y", it works in the opposite
        manner. But: For the "y" value, where the mouse points to, there might
        be multiple matching x values, if the plotted function is not biunique.
        Cursor and text coordinate will always refer to only one x value.
        So if you use the parameter value "y", ensure that your function is
        biunique.

    Other Parameters
    ----------------
    textprops : `matplotlib.text` properties as dictionary
        Specifies the appearance of the rendered text object.

    **cursorargs : `matplotlib.widgets.Cursor` properties
        Arguments passed to the internal `~matplotlib.widgets.Cursor` instance.
        The `matplotlib.axes.Axes` argument is mandatory! The parameter
        *useblit* can be set to *True* in order to achieve faster rendering.

    """

    def __init__(self, line, numberformat="{0:.4g};{1:.4g}", offset=(5, 5),
                 dataaxis='x', textprops={}, **cursorargs):
        # The line object, for which the coordinates are displayed
        self.line = line
        # The format string, on which .format() is called for creating the text
        self.numberformat = numberformat
        # Text position offset
        self.offset = np.array(offset)
        # The axis in which the cursor position is looked up
        self.dataaxis = dataaxis

        # First call baseclass constructor.
        # Draws cursor and remembers background for blitting.
        # Saves ax as class attribute.
        super().__init__(**cursorargs)

        # Default value for position of text.
        self.set_position(self.line.get_xdata()[0], self.line.get_ydata()[0])
        # Create invisible animated text
        self.text = self.ax.text(
            self.ax.get_xbound()[0],
            self.ax.get_ybound()[0],
            "0, 0",
            animated=bool(self.useblit),
            visible=False, **textprops)
        # The position at which the cursor was last drawn
        self.lastdrawnplotpoint = None

    def onmove(self, event):
        """
        Overridden draw callback for cursor. Called when moving the mouse.
        """

        # Leave method under the same conditions as in overridden method
        if self.ignore(event):
            self.lastdrawnplotpoint = None
            return
        if not self.canvas.widgetlock.available(self):
            self.lastdrawnplotpoint = None
            return

        # If the mouse left drawable area, we now make the text invisible.
        # Baseclass will redraw complete canvas after, which makes both text
        # and cursor disappear.
        if event.inaxes != self.ax:
            self.lastdrawnplotpoint = None
            self.text.set_visible(False)
            super().onmove(event)
            return

        # Get the coordinates, which should be displayed as text,
        # if the event coordinates are valid.
        plotpoint = None
        if event.xdata is not None and event.ydata is not None:
            # Get plot point related to current x position.
            # These coordinates are displayed in text.
            plotpoint = self.set_position(event.xdata, event.ydata)
            # Modify event, such that the cursor is displayed on the
            # plotted line, not at the mouse pointer,
            # if the returned plot point is valid
            if plotpoint is not None:
                event.xdata = plotpoint[0]
                event.ydata = plotpoint[1]

        # If the plotpoint is given, compare to last drawn plotpoint and
        # return if they are the same.
        # Skip even the call of the base class, because this would restore the
        # background, draw the cursor lines and would leave us the job to
        # re-draw the text.
        if plotpoint is not None and plotpoint == self.lastdrawnplotpoint:
            return

        # Baseclass redraws canvas and cursor. Due to blitting,
        # the added text is removed in this call, because the
        # background is redrawn.
        super().onmove(event)

        # Check if the display of text is still necessary.
        # If not, just return.
        # This behaviour is also cloned from the base class.
        if not self.get_active() or not self.visible:
            return

        # Draw the widget, if event coordinates are valid.
        if plotpoint is not None:
            # Update position and displayed text.
            # Position: Where the event occurred.
            # Text: Determined by set_position() method earlier
            # Position is transformed to pixel coordinates,
            # an offset is added there and this is transformed back.
            temp = [event.xdata, event.ydata]
            temp = self.ax.transData.transform(temp)
            temp = temp + self.offset
            temp = self.ax.transData.inverted().transform(temp)
            self.text.set_position(temp)
            self.text.set_text(self.numberformat.format(*plotpoint))
            self.text.set_visible(self.visible)

            # Tell base class, that we have drawn something.
            # Baseclass needs to know, that it needs to restore a clean
            # background, if the cursor leaves our figure context.
            self.needclear = True

            # Remember the recently drawn cursor position, so events for the
            # same position (mouse moves slightly between two plot points)
            # can be skipped
            self.lastdrawnplotpoint = plotpoint
        # otherwise, make text invisible
        else:
            self.text.set_visible(False)

        # Draw changes. Cannot use _update method of baseclass,
        # because it would first restore the background, which
        # is done already and is not necessary.
        if self.useblit:
            self.ax.draw_artist(self.text)
            self.canvas.blit(self.ax.bbox)
        else:
            # If blitting is deactivated, the overridden _update call made
            # by the base class immediately returned.
            # We still have to draw the changes.
            self.canvas.draw_idle()

    def set_position(self, xpos, ypos):
        """
        Finds the coordinates, which have to be shown in text.

        The behaviour depends on the *dataaxis* attribute. Function looks
        up the matching plot coordinate for the given mouse position.

        Parameters
        ----------
        xpos : float
            The current x position of the cursor in data coordinates.
            Important if *dataaxis* is set to 'x'.
        ypos : float
            The current y position of the cursor in data coordinates.
            Important if *dataaxis* is set to 'y'.

        Returns
        -------
        ret : {2D array-like, None}
            The coordinates which should be displayed.
            *None* is the fallback value.
        """

        # Get plot line data
        xdata = self.line.get_xdata()
        ydata = self.line.get_ydata()

        # The dataaxis attribute decides, in which axis we look up which cursor
        # coordinate.
        if self.dataaxis == 'x':
            pos = xpos
            data = xdata
            lim = self.ax.get_xlim()
        elif self.dataaxis == 'y':
            pos = ypos
            data = ydata
            lim = self.ax.get_ylim()
        else:
            raise ValueError(f"The data axis specifier {self.dataaxis} should "
                             f"be 'x' or 'y'")

        # If position is valid and in valid plot data range.
        if pos is not None and lim[0] <= pos <= lim[-1]:
            # Find closest x value in sorted x vector.
            # This requires the plotted data to be sorted.
            index = np.searchsorted(data, pos)
            # Return none, if this index is out of range.
            if index < 0 or index >= len(data):
                return None
            # Return plot point as tuple.
            return (xdata[index], ydata[index])

        # Return none if there is no good related point for this x position.
        return None

    def clear(self, event):
        """
        Overridden clear callback for cursor, called before drawing the figure.
        """

        # The base class saves the clean background for blitting.
        # Text and cursor are invisible,
        # until the first mouse move event occurs.
        super().clear(event)
        if self.ignore(event):
            return
        self.text.set_visible(False)

    def _update(self):
        """
        Overridden method for either blitting or drawing the widget canvas.

        Passes call to base class if blitting is activated, only.
        In other cases, one draw_idle call is enough, which is placed
        explicitly in this class (see *onmove()*).
        In that case, `~matplotlib.widgets.Cursor` is not supposed to draw
        something using this method.
        """

        if self.useblit:
            super()._update()

def load_matrix_creator(max_load_HPV,minimumViableLoad,res):
    # res = resoltuion, also used as a flag to set minimum/maxmum sinle load scearios

    if res == 1:  # if res =1 , calculate for max load of HPV
        load_matrix = np.zeros((len(max_load_HPV),res))    # initilaise numpy matrix
        load_matrix = max_load_HPV
    elif res == 0: # if res =0 , calculate for min viable load of HPV (trick: use this to set custom load)
        load_matrix = np.zeros((len(max_load_HPV),1))    # initilaise numpy matrix
        load_matrix = load_matrix+minimumViableLoad
    elif res > 1:
        load_matrix = np.zeros((len(max_load_HPV),res))    # initilaise numpy matrix

        #### Create linear space of weights
        # creates a vector for each of the HPVs, an equal number of elements spaced 
        # evenly between the minimum viable load and the maximum load for that HPV
        i=0                                                         # initliase index
        for maxval in np.nditer(max_load_HPV): # iterate through numpy array
            minval = minimumViableLoad
            load_vector =   np.linspace(start = minval,             # define linear space of weights
                            stop =maxval,                           # define maximum value for linear space
                            num = res,                              # how many data points?
                            endpoint = True,
                            retstep = False,
                            dtype = None)
            load_matrix[i:] = load_vector                           # place the vector in to a matrix
            i += 1                                                  # increment index
    else:
        print('Error: unexpected loading resolution, setting default')
        load_matrix = max_load_HPV

    return load_matrix

def max_safe_load(m_HPV_only,LoadCapacity,F_max,s,g):


    max_load_HPV = LoadCapacity  
    #### Weight limits
    # Calculate weight limits for hills. There are some heavy loads which humans will not be able to push up certain hills
    # Note that this is for average slopes etc. This will be innacurate (i.e one VERY hilly section could render the whole thing impossible, this isn't accounted for here)
    if s != 0:  #requires check to avoid divide by zero
        max_pushable_weight = F_max/(np.sin(s)*g)
        i=0
        for  HPV_weight in m_HPV_only:
            if max_load_HPV[i]+HPV_weight>max_pushable_weight:
                max_load_HPV[i] = max_pushable_weight-HPV_weight
            i+=1
    
    return max_load_HPV

def bikeModel(param_df,s,m1,P_t,F_max,L,minimumViableLoad,res):

    #### constants
    g=9.81
    pi = 3.1416
    bike_location = 2
 
    n_hpv = len(param_df.Name)
    n = np.array(param_df.Efficiency).reshape((n_hpv,1))
    Crr = np.array(param_df.Crr).reshape((n_hpv,1))
    v_no_load = np.array(param_df.AverageSpeedWithoutLoad).reshape((n_hpv,1))
    LoadCapacity = np.array(param_df.LoadCapacity).reshape((n_hpv,1))
    m_HPV_only = param_df.LoadCapacity*0.05;                    # assume 5# of load is the wight of HPV

    max_load_HPV = max_safe_load(m_HPV_only,LoadCapacity,F_max,s,g) # function to calaculate the max saffe loads due to hill
    load_matrix = load_matrix_creator(max_load_HPV[bike_location],minimumViableLoad,res)
    load_matrix = load_matrix_creator(max_load_HPV[bike_location],minimumViableLoad,res)



    #### Derivations of further mass variables
    m_HPV_pilot = np.array(m1*param_df.Pilot + m_HPV_only).reshape((n_hpv,1)) # create vector with extra weights to add
    m_HPV_load_pilot = load_matrix + m_HPV_pilot # weight of the HPV plus the rider (if any) plus the load
    m_walk_carry = m1 + m_HPV_load_pilot * (np.array(param_df.GroundContact).reshape((n_hpv,1))-1)*-1 # negative 1 is to make the wheeled = 0 and the walking = 1
    m_HPV_load = load_matrix + np.array(m_HPV_only).reshape((n_hpv,1)) 
    # weight of the mass being 'walked', i.e the wieght of the human plus anything they are carrying (not pushing or riding)

    #### Constants from polynomial equation analysis
    C = ((m_walk_carry)*g/pi)*(3*g*L/2)**(1/2)
    D = pi**2/(6*g*L)
    B1 = m_HPV_load_pilot*g*np.cos(np.arctan(s))*Crr     # rolling resistance component
    B2 = m_HPV_load_pilot * np.sin(np.arctan(s))         # slope component
    B = B1 + B2

    ##### velocities
    v_load = (-B + np.sqrt(B**2+(2*C*D*P_t)/n))/(C*D/n);    # loaded velocity

    # if loaded speed is greater than unloaded avg speed, make equal to avg unloaded speed
    i=0
    for maxval in v_no_load:
        indeces = v_load[i]>maxval; v_load[i,indeces]=maxval ; i+=1

    # calcualte average speed, half of the trip is loaded, and half is unloaded
    v_avg = (v_load+v_no_load)/2; #average velocity for a round trip

    return v_avg , load_matrix

#### Start Script
with open("data/ModelParams.csv") as csv_file:
    # read the csv file
    param_df = pd.read_csv("data/ModelParams.csv") 

#### variables (changeable)
s_deg = 0       # slope in degrees (only used for loading scenario, is overriden in cariable slope scenario)
m1=83           # mass of rider/person
P_t=75          # power output of person (steady state average)
F_max = 300     # maximum force exertion for pushing up a hill for a short amount of time
L=1             # leg length
minimumViableLoad = 15 # in kg, the minimum useful load for such a trip
t_hours=8       # number of hours to gather water






## plot options
load_plot = 1
slope_plot = 0

## Options
single_load = 0    # 1= single load, or 0 = linspace of loads
single_slope = 1    # 1= single slope, or 0 = linspace of slopes
if single_load + single_slope > 1:
    print('Sorry, can only investigate either Slope OR Loading at once')
load_res = 0        # resolution (how many datapoints for linear space)
slope_start = 0     # slope min
slope_end = 20      # slope max
slope_res = 30      # resolution (how many datapoints for linear space)

#### constants
g=9.81
pi = 3.1416

### Main call of the functions for either single or many sllope options
if single_slope == 0:
    i=0
    slope_vector =   np.linspace(start = slope_start, stop =slope_end, num = slope_res)     
    d_var_matrix = np.zeros((len(slope_vector),len(param_df.Name)))
    v_avg_matrix = np.zeros((len(slope_vector),len(param_df.Name)))
    load_vector_matrix = np.zeros((len(slope_vector),len(param_df.Name)))

    for slope in slope_vector:
        s =  (slope/360)*(2*pi)
        v_avg , load_matrix = walkmodel(param_df,s,m1,P_t,F_max,L,minimumViableLoad,load_res)
        v_avg_matrix[i] = v_avg.reshape(len(param_df.Name))
        load_vector_matrix[i] = load_matrix.reshape(len(param_df.Name))
        i+=1
else:
    s = (s_deg/360)*(2*pi) #converts s in to radians
    v_avg , load_matrix = bikeModel(param_df,s,m1,P_t,F_max,L,minimumViableLoad,load_res)

# calculation of velocity kgs
velocitykgs=v_avg_matrix*load_vector_matrix

# #### Forces and Power
# F_push_hill = np.sin(s)*m_HPV_load*g # force required to push uphill (approx)

#### Distance
t_secs=t_hours*60*60
distance_achievable = (v_avg*t_secs)/1000 #kms

# np.savetxt("velocitykgs.csv", velocitykgs, delimiter=",")

# # open a file, where you ant to store the data
# file = open('outputvals.pkl', 'wb')
# pickle.dump([v_avg_matrix, load_vector_matrix, slope_vector] , file)
# file.close()

if slope_plot == 1:
    i=0

#   # Slope Graph Sensitivity
    fig, ax = plt.subplots(figsize=(20, 10))
    for HPVname in param_df.Name:
        
        x = slope_vector
        y = v_avg_matrix[:,i]*load_vector_matrix[:,i]
        line, = ax.plot(x, y, label=HPVname)  # Plot some data on the axes.
        i += 1
    plt.xlabel('Slope [deg ˚]')
    plt.ylabel('Velocity Kgs [m/s]')
    plt.title("Velocity Kgs as a function of Slope Plot")
    plt.legend();


    cursor = AnnotatedCursor(
        line=line,
        numberformat="{0:.2f}\n{1:.2f}",
        dataaxis='x', offset=[10, 10],
        textprops={'color': 'blue', 'fontweight': 'bold'},
        ax=ax,
        useblit=True,
        color='red', linewidth=2)

    plt.show()

elif load_plot ==1:
    i=0
# #   # Load Graph Sensitivity
#     fig, ax = plt.subplots(figsize=(20, 10))
#     for HPVname in param_df.Name:
#         ax.plot(load_matrix[i], v_avg[i]*load_matrix[i], label=HPVname)  # Plot some data on the axes.
#         i += 1
#     plt.xlabel('Load [kg]')
#     plt.ylabel('Velocity Kgs [m/s]')
#     plt.title("Simple Plot")
#     plt.legend();

#   # Slope Graph Sensitivity
    fig, ax = plt.subplots(figsize=(20, 10))
    for HPVname in param_df.Name:
        
        x = slope_vector
        y = v_avg_matrix[:,i]
        line, = ax.plot(x, y, label=HPVname)  # Plot some data on the axes.
        i += 1
    plt.xlabel('Slope [deg ˚]')
    plt.ylabel('Velocity [m/s]')
    plt.title("Velocity as a function of Slope Plot with constant load")
    plt.legend();


    cursor = AnnotatedCursor(
        line=line,
        numberformat="{0:.2f}\n{1:.2f}",
        dataaxis='x', offset=[10, 10],
        textprops={'color': 'blue', 'fontweight': 'bold'},
        ax=ax,
        useblit=True,
        color='red', linewidth=2)

    plt.show()






    # #### Plotting
    # i=0
    # fig = plt.figure(), #ax = plt.subplots(figsize=(20, 10))
    # ax = fig.add_subplot(projection='3d')
    # for HPVname in param_df.Name:
    #     # ax.plot(load_vector_matrix[0:10,i], v_avg_matrix[0:10,i], label=HPVname)  # Plot some data on the axes.
    #     ax.scatter(load_vector_matrix[:,i], v_avg_matrix[:,i], slope_vector,  label=HPVname)
        
    #     i += 1
    # plt.xlabel('Load [kg]')
    # plt.ylabel('Velocity [m/s]')
    # plt.title("Simple Plot")
    # plt.legend();

    # filename_output = 'velocity_slope' + str(s_deg)
    # fig.savefig(filename_output, transparent=False, dpi=80, bbox_inches="tight")

