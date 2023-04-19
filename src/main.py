"""
@author: Prabhat Ale
"""

import streamlit as st
import plotly.io as pio
import config as cfg
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

def display_props():
	# header
	st.markdown("####  Embeddings explorer with different Reduction Schemes and Search Functionality")
	# feature image
	image = Image.open('images/fusemachine.png')
	st.sidebar.image(image, use_column_width=True)
	return


def load_data(file):

	df = pd.read_table(file, sep=' ')
	data = df.values.tolist()
	labels = [d[0] for d in data]
	index = [d[1] for d in data]
	data = np.array([d[2:] for d in data])
	return data, labels,index


	
## dimension reductions
def display_reductions():
	reductions = ("PCA", "TSNE")
	options = list(range(len(reductions)))
	reductions_type = st.sidebar.selectbox("Select Dim. Reduction", options, format_func=lambda x: reductions[x])
	return reductions_type

# no. dimensions
def display_dimensions():
	dims = ("2-D", "3-D")
	dim = st.sidebar.radio("Dimensions", dims)	
	return dim


def plot_2d_updated(df, labels, need_labels,index, search=None):

	df['index']=index
	unique_labels = np.unique(labels)	
	if not need_labels:
		labels=None
	df['labels'] = labels
	if search: 
		fig=plot_with_opacity(df, search)
	else:
		fig = px.scatter(df, x='pc1', y='pc2',
					color='labels', color_discrete_sequence=cfg.colors[:len(unique_labels)],
					hover_name='index')
	return fig


def plot_with_opacity(df, search):
	search = search.split(',')
	search = [int(item) for item in search]
	print(search)
	# Build figure
	fig = go.Figure()
	print(df.loc[df['index'].isin(search), 'pc1'].shape[0])
	# Add first scatter trace with medium sized markers
	fig.add_trace(
		go.Scatter(
			mode='markers',
			x=df.loc[df['index'].isin(search), 'pc1'],
			y=df.loc[df['index'].isin(search), 'pc2'],
			opacity=1,
			marker=dict(
				color='LightSkyBlue',
				size=30,
				line=dict(
					color='MediumPurple',
					width=2
				)
			),
			name='Opacity 1.0'
		)
	)

	fig.add_trace(
		go.Scatter(
			mode='markers',
			x=df.loc[~df['index'].isin(search), 'pc1'],
			y=df.loc[~df['index'].isin(search), 'pc2'],
			opacity=0.2,
			marker=dict(
				color='LightSkyBlue',
				size=5,
				line=dict(
					color='MediumPurple',
					width=2
				)
			),
			name='Opacity 0.2'
		)
	)
	return fig




	# fig = px.scatter(df, x='pc1', y='pc2',
	# 			color='labels', color_discrete_sequence=cfg.colors[:len(unique_labels)],
	# 			hover_name='index')
	# return fig


	
def plot_3d_updated(df, labels, need_labels,index, search=None):
	 
	sizes = [5]*len(labels)
	df['index']=index
	df['labels'] = labels
	if search: 
		sizes[search] = 25
	unique_labels = np.unique(labels)	
	if not need_labels:
		labels=None
	fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3',
				color='labels', color_discrete_sequence=cfg.colors[:len(unique_labels)],
				hover_name='index')
	return fig

	

# search
def display_search():
	search_for = st.sidebar.text_input("Label Lookup", "")
	return search_for

#labels check
def display_labels():
	need_labels = st.sidebar.checkbox("Display Labels", value=True)
	return need_labels

def display_animation():
	need_animation = st.sidebar.checkbox("Display Animation", value=True)
	return need_animation


def rotate_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

def render_plot_2d(fig):
	fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=800)
	fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 20)))
	fig.update_traces(marker_size = 5)
	st.plotly_chart(fig)

def render_plot_3d(fig, need_animation):
	fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=800)
	fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 20)))
	fig.update_traces(marker_size = 5)
	st.plotly_chart(fig)
	if not need_animation:
		need_animation=None
	else:
		# Use for animation rotation at the end
		x_eye = -1.25
		y_eye = 2
		z_eye = 0.5
		# Use for animation rotation
		fig.update_layout(scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
					updatemenus=[dict(type='buttons',
										showactive=False,
										y=1,
										x=0.8,
										xanchor='left',
										yanchor='bottom',
										pad=dict(t=45, r=10),
										buttons=[dict(label='Play',
													method='animate',
													args=[None, dict(frame=dict(duration=250, redraw=True),
																	transition=dict(duration=0),
																	fromcurrent=True,
																	mode='immediate', uirevision=True
																	)]
													)
												]
										)
								]
					)
		frames = []
		for t in np.arange(0, 6.26, 0.1):
			xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
			frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
		fig.frames = frames
		pio.show(fig)




def plot_for_2D(data, labels, need_labels, index, search_idx=None):

	fig = plot_2d_updated(data, labels, need_labels,index, search_idx)
	render_plot_2d(fig)


def plot_for_3D(data, labels, need_labels, index, need_animation, search_idx=None):

	fig = plot_3d_updated(data, labels, need_labels, index,  search_idx)
	render_plot_3d(fig, need_animation)




if __name__ == "__main__":
	display_props()
	uploaded_file = st.sidebar.file_uploader("Upload a file (Optional)", type="txt")
	reductions_type = display_reductions()
	dim = display_dimensions()
	search_for = display_search()
	need_labels = display_labels()
	need_animation = display_animation()
	button = st.sidebar.button('Visualise')
	if uploaded_file:
		data, labels, indexes = load_data(uploaded_file)
		if button:
			if dim=='2-D':
				pca = PCA(n_components=2)
				data = pca.fit_transform(data)
				pca_df = pd.DataFrame(data=data, columns=['pc1', 'pc2'])
				if search_for:
					search_idx = search_for
					# search_idx = indexes.index(int(search_for))
					plot_for_2D(pca_df, labels, need_labels, indexes, search_idx)
				else:
					plot_for_2D(pca_df, labels, need_labels,indexes)

			else:
				pca = PCA(n_components=3)
				data = pca.fit_transform(data)
				pca_df = pd.DataFrame(data=data, columns=['pc1', 'pc2', 'pc3'])
				if search_for:
					search_idx = labels.index(search_for)
					plot_for_3D(pca_df, labels, need_labels, indexes, need_animation, search_idx)
				else:
					plot_for_3D(pca_df, labels, need_labels, indexes, need_animation)



