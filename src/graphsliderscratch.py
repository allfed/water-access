import plotly.graph_objects as go 
from ipywidgets import interact
import numpy as np

def linear_function(a,b,x):
    return a*x+b

a0 = 0.1
b0 = 0.1
x = np.arange(-10,10,0.1)

data = [go.Scatter(x=x,y=linear_function(a=a0,b=b0,x=x))]

fig = go.FigureWidget(data=data)
fig.update_yaxes(autorange=False,range=(-3,3))

@interact(a=(0, 1, 0.1), b=(0, 1, 0.1))
def update(a=a0, b=b0):
    with fig.batch_update():
        fig.data[0].y = linear_function(a=a,b=b,x=x)
    return fig