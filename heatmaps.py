import plotly as py
# py.offline.init_notebook_mode()
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from numpy import genfromtxt

activations = genfromtxt('rnn_activations_pred',delimiter=',')
activations = activations.T
print activations[0][0:30]
print activations.shape
activations = np.reshape(activations,(100,30,30))
print activations[0][0]


gen_str = open('pred_feature_1.0_1.txt').read()
gen_str = gen_str[24:-1]
print len(gen_str)

data = ff.create_annotated_heatmap(
        z=activations[13],
        x=range(activations.shape[1]),
        y=range(activations.shape[1]),
        annotation_text=np.reshape(np.array(list(gen_str)),(30,30)),
        text = np.reshape(np.array(list(gen_str)),(30,30))
    )
layout = go.Layout(
    autosize=False,
    width=500,
    height=1000,
)
py.offline.iplot(data,layout)