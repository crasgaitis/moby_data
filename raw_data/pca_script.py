import pickle
import numpy as np 
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from tqdm import tqdm

from oorcas import HydrophoneDay
from datetime import timedelta, datetime
from collections import Counter

FILL_VALUE=None
METHOD=1
max_streams = 50

hyd_days = []

start_date = datetime(2024, 8, 23)
end_date = datetime(2024, 8, 25)

current_date = start_date
while current_date <= end_date:
    hyd_days.append(current_date.strftime("%Y/%m/%d"))
    current_date += timedelta(days=1)

print(hyd_days)

for hyd_day in tqdm(hyd_days):

    hyd = HydrophoneDay("CE04OSBP-LJ01C-11-HYDBBA105", str(hyd_day))

    hyd.read_and_repair_gaps(fill_value=FILL_VALUE, method=METHOD, wav_data_subtype="PCM_32")

    data_matrix = []
    for stream_index, stream in (enumerate(hyd.clean_list[:max_streams])):
        for trace_index, trace in enumerate(stream.traces):
            trace_data = trace.data[::100]
            if len(trace_data == 192000):
                data_matrix.append(trace_data)

    lengths = [len(item) for item in data_matrix]
    length_counts = Counter(lengths)

    max_length = max(length_counts.keys())
    filtered_data_matrix = [item for item in data_matrix if len(item) == max_length]
    data_matrix = np.array(filtered_data_matrix)

    n_components = 10
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_matrix)

    save_file = str(hyd_day.replace('/', '_')) + "_pca_data.pkl"

    with open(save_file, 'wb') as f:
        pickle.dump({
            'data_matrix': data_matrix,
            'pca': pca,
            'data_pca': data_pca
        }, f)

    print("Data saved successfully as pca_data.pkl")