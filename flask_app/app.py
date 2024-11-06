import wave
import librosa
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from oorcas import HydrophoneDay
import xarray as xr
import base64
from io import BytesIO
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from obspy import read
from flask import Flask, render_template, request, send_file
from matplotlib import pyplot as plt
from scipy import signal
import pickle

app = Flask(__name__)

win = "hann"
L = 4096
overlap = 0.5
n_clusters = 4  
detection_params = {'click_duration': 0.01, 'freq_min': 2000, 'freq_max': 25000, 'intensity_threshold': 0.8}

with open('../raw_data/pca_data.pkl', 'rb') as f:
    data = pickle.load(f)

data_matrix = data['data_matrix']
pca = data['pca']
data_pca = data['data_pca']

# utils
def load_user_data(file_path, file_type):
    global data, sample_rate
    if file_type == 'wav':
        sample_rate, data = wavfile.read(file_path)
    elif file_type == 'mseed':
        st = read(file_path)
        tr = st[0]  # assuming single trace
        sample_rate, data = tr.stats.sampling_rate, tr.data
    return sample_rate, data
        
def plot_spectrogram(data, fs, nperseg=256):
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=nperseg)
    
    Sxx_subsampled = Sxx[::10, ::10]
    f_subsampled = f[::10]
    t_subsampled = t[::10]

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(t_subsampled, f_subsampled, 10 * np.log10(Sxx_subsampled), shading='gouraud')
    plt.title('Subsampled Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Intensity [dB]')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def plot_spectrogram_heatmap(data, fs, win="hann", L=4096, overlap=0.5, scale="log"):

    f, t, Sxx = signal.spectrogram(
        x=data,
        fs=fs,
        window=win,
        nperseg=L,
        noverlap=int(L * overlap),
        scaling='density'
    )

    if scale == "log":
        Sxx = 10 * np.log10(Sxx)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(Sxx, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()],
               cmap='inferno', interpolation='nearest')
    plt.colorbar(label='PSD [dB re 1µPa^2 / Hz]')
    plt.title('Spectrogram Heatmap')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.ylim(f.min(), f.max())
    plt.xlim(t.min(), t.max())
    plt.grid(False)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def compute_psd_welch_and_plot(data, fs, win="hann", L=4096, overlap=0.5, avg_method="median", 
                                interpolate=None, scale="log"):
    nfft = L if interpolate is None else int(fs / interpolate) if fs / L > interpolate else L

    f, Pxx = signal.welch(
        x=data,
        fs=fs,
        window=win,
        nperseg=L,
        noverlap=int(L * overlap),
        nfft=nfft,
        average=avg_method,
    )

    if scale == "log":
        Pxx = 10 * np.log10(Pxx)
    
    psd_xr = xr.DataArray(
        np.array(Pxx),
        dims=["frequency"],
        coords={"frequency": np.array(f)},
        attrs=dict(
            nperseg=L,
            units="dB re µPa^2 / Hz",
        ),
        name="psd",
    )

    f_subsampled = f[::10]
    Pxx_subsampled = Pxx[::10]

    plt.figure(figsize=(8, 6))
    plt.plot(f_subsampled, Pxx_subsampled)
    plt.title('Subsampled Power Spectral Density')
    plt.ylabel('PSD [dB re 1µPa^2 / Hz]')
    plt.xlabel('Frequency [Hz]')
    plt.xscale('log') 
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return psd_xr, image_base64

def plot_complex_spectrogram(f, t, Zxx):

    plt.figure(figsize=(8, 6))
    plt.imshow(20 * np.log10(np.abs(Zxx)), aspect='auto', origin='lower', 
               extent=[t.min(), t.max(), f.min(), f.max()],
               cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Magnitude [dB]')
    plt.title('Magnitude Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.ylim(f.min(), f.max())
    plt.xlim(t.min(), t.max())
    plt.grid(False)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def compute_clustering_plot(data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_pca)
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
    labels = kmeans.labels_

    new_sample = np.array(data[::1000]).reshape(1, -1)
    new_sample_pca = pca.transform(new_sample)

    predicted_cluster = kmeans.predict(new_sample_pca)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
       
    for i in range(n_clusters):
        ax.scatter(data_pca[labels == i, 0], data_pca[labels == i, 1], 
                   data_pca[labels == i, 2], color=colors[i], label=f'Cluster {i}')
     
    ax.scatter(new_sample_pca[0, 0], new_sample_pca[0, 1], new_sample_pca[0, 2], 
            color='red', marker='*', s=200, label='Sample', edgecolor=colors[predicted_cluster])
    
    ax.set_title('KMeans Clustering Results (3D PCA)')

    ax.legend() 
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64, labels

def process_clusters(labels, sample_rate, win, L, overlap):
    instances = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            instances.append(data_matrix[cluster_indices[0]])

    spectrogram_images = []
    for instance in instances:
        f, t, Zxx = signal.stft(
            x=instance,
            fs=sample_rate,
            window=win,
            nperseg=L,
            noverlap=int(L * overlap),
            boundary=None,
            return_onesided=False
        )
        
        image_base64 = plot_complex_spectrogram(f, t, Zxx)
        spectrogram_images.append(image_base64)

    return spectrogram_images

def detect_whale_clicks():

    click_duration = detection_params['click_duration']
    freq_min = detection_params['freq_min']
    freq_max = detection_params['freq_max']
    intensity_threshold = detection_params['intensity_threshold']
    
    click_samples = int(click_duration * sample_rate)
    
    sos = signal.butter(4, [freq_min, freq_max], btype='band', fs=sample_rate, output='sos')
    print('sos')
    filtered_data = signal.sosfilt(sos, data)
    print('filter')
    
    envelope = np.abs(signal.hilbert(filtered_data))
    print('envelope')
    print(envelope)
    print(np.max(envelope))
    clicks = envelope > intensity_threshold * np.max(envelope)
    print(clicks)
    click_starts = np.where(np.diff(clicks.astype(int)) == 1)[0]
    click_ends = np.where(np.diff(clicks.astype(int)) == -1)[0]
    
    if len(click_starts) > len(click_ends):
        click_starts = click_starts[:-1]
    elif len(click_ends) > len(click_starts):
        click_ends = click_ends[1:]
    
    detected_clicks = []
    for start, end in zip(click_starts, click_ends):
        if (end - start) >= click_samples:
            detected_clicks.append(data[start:end])
    
    return detected_clicks


# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Explore page route
@app.route('/explore')
def explore():
    return render_template('explore.html')

# Upload page route
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file(upload_folder="flask_app/app.py/upload_folder"):
    global file_path, detection_params, spectrogram_image, spectrogram_heatmap, psd_plot, complex_spectrogram, cluster_info, cluster1, cluster2, cluster3, cluster4, detection_params
    print(request.files)
    if 'file' in request.files:
        if request.method == 'POST':
            file = request.files['file']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            if file and (file.filename.endswith('.wav') or file.filename.endswith('.mseed')):
                file_type = 'wav' if file.filename.endswith('.wav') else 'mseed'
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                # after upload, downstream plot/data generation  
                sample_rate, data = load_user_data(file_path, file_type)  
                        
                spectrogram_image = plot_spectrogram(data, sample_rate)
                spectrogram_heatmap = plot_spectrogram_heatmap(data, sample_rate)
                psd_xr, psd_plot = compute_psd_welch_and_plot(data, sample_rate)
                
                f, t, Zxx = signal.stft(
                    x=data,
                    fs=sample_rate,
                    window=win,
                    nperseg=L,
                    noverlap=int(L * overlap),
                    boundary=None,
                    return_onesided=False
                )
                
                complex_spectrogram = plot_complex_spectrogram(f, t, Zxx)
                
                cluster_info, labels = compute_clustering_plot(data)
                cluster1, cluster2, cluster3, cluster4 = process_clusters(labels, sample_rate, win, L, overlap)
                
                return render_template('upload.html', spectrogram_image=spectrogram_image,
                                    spectrogram_heatmap = spectrogram_heatmap,
                                    psd_plot=psd_plot, complex_spectrogram=complex_spectrogram,
                                    cluster_info=cluster_info,
                                    cluster1=cluster1, cluster2=cluster2, cluster3=cluster3, cluster4=cluster4,
                                    detection_params=detection_params)
                
        return render_template('upload.html')
    
    else:
        detection_params['click_duration'] = float(request.form['click_duration'])
        detection_params['freq_min'] = float(request.form['freq_min'])
        detection_params['freq_max'] = float(request.form['freq_max'])
        detection_params['intensity_threshold'] = float(request.form['intensity_threshold'])

        if file_path:
            print('yo')
            
            click_times = detect_whale_clicks()
            print(click_times)
            
            return render_template('upload.html', spectrogram_image=spectrogram_image,
                                        spectrogram_heatmap = spectrogram_heatmap,
                                        psd_plot=psd_plot, complex_spectrogram=complex_spectrogram,
                                        cluster_info=cluster_info,
                                        cluster1=cluster1, cluster2=cluster2, cluster3=cluster3, cluster4=cluster4, 
                                        click_times=click_times, detection_params=detection_params)


if __name__ == '__main__':
    app.run(debug=True)