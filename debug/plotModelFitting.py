# This Python file uses the following encoding: utf-8
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
import numpy as np 
import plotly.io as pio
pio.renderers.default = 'browser'
class plotModelFitting:
    def __init__(self):
        pass

    @staticmethod
    def plot_signal_with_fiducials_interactive(signal, time, sys, val, dic_notches, trend, waves, model, idx_misfit, R2, title="Skew gaussian model"):
        """
        Plot interactive signal with systolic peaks and diastolic valleys using Plotly.

        Parameters:
        - signal: 1D array of the filtered signal
        - time: 1D array of time values
        - sys: systolic peaks (Nx3) array [index, time, amplitude]
        - val: valley points (Mx3) array [index, time, amplitude]

        Returns:
        - fig: plotly figure object
        """
        fig = make_subplots(rows=2, cols=1)

        # Plot main signal
        fig.add_trace(go.Scatter(
            x=time, y=signal,
            mode='lines',
            name='Signal',
            line=dict(color='black')
        ), row=1,col=1)

        # Plot systolic peaks
        if sys is not None and len(sys) > 0:
            fig.add_trace(go.Scatter(
                x=sys[:, 1], y=sys[:, 2],
                mode='markers',
                name='Systolic Peaks',
                marker=dict(color='red', size=8, symbol='circle')
            ),row=1,col=1)

        # Plot valleys
        if val is not None and len(val) > 0:
            fig.add_trace(go.Scatter(
                x=val[:, 1], y=val[:, 2],
                mode='markers',
                name='Diastolic Valleys',
                marker=dict(color='green', size=8, symbol='square')
            ),row=1,col=1)

        if dic_notches is not None:
            fig.add_trace(go.Scatter(
            x=dic_notches[:, 0], y=dic_notches[:, 1],
            mode='markers',
            name="Dicrotic Notch",
            marker=dict(color="cyan", size=8, symbol="triangle-up")
            ),row=1,col=1)

        if trend is not None:
            fig.add_trace(go.Scatter(
            x=time, y=trend, name="Trend",
            line=dict(color="gray", dash="dash")
            ),row=1,col=1)
            
        # Component waves
        colors = ['red', 'green', 'orange']
        for i in range(3):
            if waves is not None:
                fig.add_trace(go.Scatter(
                    x=time,
                    y=waves[:, i]+trend,
                    mode='lines',
                    name=f'Wave {i + 1}',
                    line=dict(color=colors[i], dash='dot')
                ), row=1, col=1)
                
        # Plot main signal
        fig.add_trace(go.Scatter(
                x=time, y=signal,
                mode='lines',
                name='Signal',
                line=dict(color='black')
            ), row=2,col=1)     
            
        if model is not None:
            fig.add_trace(go.Scatter(
                x=time, y=model + trend,
                mode='lines',
                name='Model',
                line=dict(color='red')
            ), row=2, col=1)
    

        fig.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            height=500,
            width=900,
            template='simple_white'
        )
        fig.show()
        return fig
