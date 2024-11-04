import matplotlib.pyplot as plt
import os
import fsspec
import numpy as np 
from sklearn.decomposition import PCA

from oorcas import HydrophoneDay
from datetime import timedelta, datetime
from pathlib import Path 
from obspy import Trace, Stream, UTCDateTime

from IPython.display import Audio

FILL_VALUE=None
METHOD=1
max_streams = 50


hyd_days = []

hyd = HydrophoneDay("CE04OSBP-LJ01C-11-HYDBBA105", "2024/02/11")

hyd.read_and_repair_gaps(fill_value=FILL_VALUE, method=METHOD, wav_data_subtype="PCM_32")

data_matrix = []
for stream_index, stream in (enumerate(hyd.clean_list[:max_streams])):
    for trace_index, trace in enumerate(stream.traces):
        data_matrix.append(trace.data[::1000])       # TODO: remove ::1000 when needed

data_matrix = np.array(data_matrix)


pca = PCA(n_components=10)
pca_result = pca.fit_transform(data_matrix)

