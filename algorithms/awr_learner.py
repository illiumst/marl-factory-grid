from common import BaseLearner, TrajectoryBuffer


class AWRLearner(BaseLearner):
    def __init__(self, *args,  buffer_size=1e5, **kwargs):
        super(AWRLearner, self).__init__(*args, **kwargs)
        assert self.train_every[0] == 'episode', 'AWR only supports the episodic RL setting!'
        self.buffer = TrajectoryBuffer(buffer_size)

    def train(self):
        # convert to trajectory format
        pass

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.25, rc={'text.usetex': True})
data = np.array([[689, 74], [71, 647]])
cats = ['Mask', 'No Mask']
df = pd.DataFrame(data/np.sum(data), index=cats, columns=cats)

group_counts = ['{0:0.0f}'.format(value) for value in
                data.flatten()]
group_percentages = [f'{value*100:.2f}' + r'$\%$' for value in
                     data.flatten()/np.sum(data)]

labels = [f'{v1}\n{v2}' for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

with sns.axes_style("white"):
    cmap = sns.diverging_palette(h_neg=100, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
    sns.heatmap(data, annot=labels, fmt='', cmap='Set2_r', square=True, cbar=False, xticklabels=cats,yticklabels=cats)
plt.title('Simple-CNN')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('cnn.pdf', bbox_inches='tight')