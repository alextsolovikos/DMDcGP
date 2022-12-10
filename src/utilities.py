import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_3d_iso_lcss(t, field, X, Y, Z, isomin=0.05, isomax=0.1,
                     w=1024, h=720, positive=True, negative=True, 
                     eye=dict(x=0, y=1.5, z=1.0), center=dict(x=0, y=0, z=0),
                     up=dict(x=0, y=1, z=0), save_as=None, mid_plane=False, show=True, 
                     jet_gain=None, **args):

    data = []
    if negative:
        data.append(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=-field[t,:,:,:,0].flatten(),
                isomin=-isomax,
                isomax=-isomin,
                cmin=-0.04,
                cmax=0.04,
                colorscale='RdBu',
                **args
            ))

    if positive:
        data.append(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=-field[t,:,:,:,0].flatten(),
                isomin=isomin,
                isomax=isomax,
                cmin=-0.04,
                cmax=0.04,
                colorscale='RdBu',
                **args
            ))
    
    if jet_gain is not None:
        data.append(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=jet_gain.flatten(),
                isomin=0.015,
                isomax=0.015,
                cmin=-0.015,
                cmax=0.015,
                colorscale='RdGy',
                opacity=0.2,
                surface_count=1
            ))
        
        
    if mid_plane:
        const_black = [[0, '#000000'], [1, '#000000']]
        data.append(
            go.Surface(
                x=34.2*np.ones(11), y=np.linspace(0.2,1.4,11), z=[np.linspace(1.5,3.5,11)]*11, colorscale=const_black, opacity=0.5,  showscale=False
        ))
        
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            width=w,
            height=h,
            scene = dict(
                camera=dict(
                    eye=eye,
                    center=center,
                    up=up
                ), #the default values are 1.25, 1.25, 1.25
                xaxis=dict(
                    title="Streamwise",
                    titlefont=dict(size=28),
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    title='',
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                ),
                zaxis=dict(
                    title="Spanwise",
                    titlefont=dict(size=28),
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                aspectmode='data', #this string can be 'data', 'cube', 'auto', 'manual'
                aspectratio=dict(x=1, y=1, z=0.95),
                bgcolor='white',
            )
        )
    )
    
    fig.update_layout(
        legend_title="u_x'",
        font=dict(
            size=18,
    ))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_traces(showscale=False)
    
    if show:
        fig.show()
    
    if save_as:
        fig.write_image(save_as, width=1.5*w, height=1.5*h)
