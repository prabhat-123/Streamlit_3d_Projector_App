"""
@author: Prabhat Ale
"""

import numpy as np
import pandas as pd
import config as cfg
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.decomposition import PCA


def load_data(file):

    df = pd.read_table(file, sep=" ")
    data = df.values.tolist()
    labels = [d[0] for d in data]
    index = [d[1] for d in data]
    data = np.array([d[2:] for d in data])
    return data, labels, index


def plot_3d_updated(df, labels, index):

    df["index"] = index
    df["labels"] = labels
    unique_labels = np.unique(labels)
    fig = px.scatter_3d(
        df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="labels",
        color_discrete_sequence=cfg.colors[: len(unique_labels)],
        hover_name="index",
    )
    return fig


def rotate_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def render_plot_3d(fig):
    fig.update_layout(margin={"r": 50, "t": 100, "l": 0, "b": 0}, height=800)
    fig.update_layout(
        legend=dict(title_font_family="Times New Roman", font=dict(size=20))
    )
    fig.update_traces(marker_size=5)
    # Use for animation rotation at the end
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    # Use for animation rotation
    fig.update_layout(
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=250, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                                uirevision=True,
                            ),
                        ],
                    )
                ],
            )
        ]
    )
    frames = []
    for t in np.arange(0, 6.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze)))
        )
    fig.frames = frames
    st.plotly_chart(fig)



def plot_for_3D(data, labels, index):

    fig = plot_3d_updated(data, labels, index)
    render_plot_3d(fig)


if __name__ == "__main__":

    data, labels, indexes = load_data('data/data_2500.txt')
    pca = PCA(n_components=3)
    data = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=data, columns=["pc1", "pc2", "pc3"])
    plot_for_3D(
        pca_df, labels, indexes)
