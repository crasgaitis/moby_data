"""
@jdduprey
Converts OOI hydrophone data stored as mseed files on the OOI raw data archive 
into 5 minute wav files using obspy and soundfile. Wav file names are written to "./acoustic/wav/YYYY_MM_DD".
Files are named in the datetime format "YYMMDDHHMMSS"
The user can set the following processing parameters: 

HYD_REFDES
    The OOI reference designator for the hydrophone you want to process. For example, 
    "CE04OSBP-LJ01C-11-HYDBBA105" is the OOI hydrophone at the Oregon Offshore (600m) site. 
    "CE04OSBP-LJ01C-11-HYDBBA110" is the co-located Ocean Sonics test hydrophone at that same site.
DATE
    The day of hydrophone data you would like to convert to wav in the date format
    YYYY/MM/DD.
FILL_VALUE
    The value obspy will use to fill any gaps within an mseed file greater than 0.02 seconds. (edge case).
    See obspy docs: https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.merge.html
METHOD
    The method obspy will use to handle data "traces" that have an overlap greater that 0.02 seconds. (edge case).
    https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html#obspy.core.stream.Stream._cleanup
SR
    Sample rate you wish to use when saving wav files. OOI Hydrophone sampling rate is 64000 Hz.
WAV_DATA_SUBTYPE
    'PCM_32' or 'FLOAT' The data subtype format for the resulting WAV files. OOI data is int32, 
     but some media players cannot import in this format. See `sf.available_subtypes('WAV')`
NORMALIZE_TRACES
    Option to normalize signal by mean of each 5 minute trace. If normalized float32 data type is needed.
"""

import fsspec
import concurrent.futures
import obspy as obs
import numpy as np
import multiprocessing as mp
import soundfile as sf

from datetime import datetime
from tqdm import tqdm
from pathlib import Path

def _map_concurrency(func, iterator, args=(), max_workers=-1, verbose=False):
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(func, i, *args): i for i in iterator}
        # Disable progress bar
        is_disabled = not verbose
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url), total=len(iterator), disable=is_disabled
        ):
            data = future.result()
            results.append(data)
    return results


class HydrophoneDay:

    def __init__(
        self,
        refdes,
        str_date,
        data=None,
        mseed_urls=None,
        clean_list=None,
        stream=None,
        spec=None,
    ):
        self.refdes = refdes
        self.date = datetime.strptime(str_date, "%Y/%m/%d")
        self.data = data
        self.mseed_urls = self.get_mseed_urls(str_date, refdes)
        self.clean_list=clean_list
        self.stream=stream
        self.spec=spec
        self.file_str = f"{self.refdes}_{self.date.strftime('%Y_%m_%d')}"


    def get_mseed_urls(self, day_str, refdes):

        base_url = "https://rawdata.oceanobservatories.org/files"
        mainurl = f"{base_url}/{refdes[0:8]}/{refdes[9:14]}/{refdes[15:27]}/{day_str}/"
        FS = fsspec.filesystem("http")
        print(mainurl)
    
        try:
            data_url_list = sorted(
                f["name"]
                for f in FS.ls(mainurl)
                if f["type"] == "file" and f["name"].endswith(".mseed")
            )
        except Exception as e:
            print("Client response: ", e)
            return None
    
        if not data_url_list:
            print("No Data Available for Specified Time")
            return None
    
        return data_url_list

    
    def read_and_repair_gaps(self, fill_value, method, wav_data_subtype):
        self.clean_list = _map_concurrency(
            func=self._deal_with_gaps_and_overlaps, 
            args=(fill_value, method, wav_data_subtype), 
            iterator=self.mseed_urls, verbose=False
        )
        
            
    def _merge_by_timestamps(self, st):
        cs = st.copy()
        
        data = []
        for tr in cs:
            data.append(tr.data)
        data_cat = np.concatenate(data)
    
        stats = dict(cs[0].stats)
        stats["starttime"] = st[0].stats["starttime"]
        stats["endtime"] = st[-1].stats["endtime"]
        stats["npts"] = len(data_cat)
    
        cs = obs.Stream(traces=obs.Trace(data_cat, header=stats))
    
        return cs
        

    def _deal_with_gaps_and_overlaps(self, url, fill_value, method, wav_data_subtype):
        if wav_data_subtype not in ["PCM_32", "FLOAT"]:
            raise ValueError("Invalid wav data subtype. Please specify 'PCM_32' or 'FLOAT'")
        # first read in mseed
        if wav_data_subtype == "PCM_32":
            st = obs.read(url, apply_calib=False, dtype=np.int32)
        if wav_data_subtype == "FLOAT":
            st = obs.read(url, apply_calib=False, dtype=np.float64)
        
        
        trace_id = st[0].stats["starttime"]
        print("total traces before concatenation: " + str(len(st)), flush=True)
        # if 19.2 samples +- 640 then concat
        samples = 0
        for trace in (st):
            samples += len(trace)
            
        if 19199360 <= samples <= 19200640: # CASE A: just jitter, no true gaps
            print(f"There are {samples} samples in this stream, Simply concatenating")
            cs = self._merge_by_timestamps(st)
            print("total traces after concatenation: " + str(len(cs)))
        else:
            print(f"{trace_id}: there are a unexpected number of samples in this file. Checking for large gaps:")
            gaps = st.get_gaps()
            st_contains_large_gap = False
            # loop checks for large gaps
            for gap in gaps:
                if abs(gap[6]) > 0.02: # the gaps 6th element is the gap length 
                    st_contains_large_gap = True
                    break
            
            if st_contains_large_gap: # CASE B: - edge case - large gaps that should be filled using obspy fill_value and method of choice
                print(f"{trace_id}: there is a gap not caused by jitter. Using obspy method={method}, fill_value={str(fill_value)}")
                cs = st.merge(method=method, fill_value=fill_value)
                print("total trace after merge: " + str(len(cs)))
            else: # CASE C: shortened trace before divert with no large gaps
                print(f"{trace_id}: This file is short but only contains jitter. Simply concatenating")
                cs = self._merge_by_timestamps(st)
                print("total traces after concatenation: " + str(len(cs)), flush=True)
        return cs
    
def convert_mseed_to_wav(
    hyd_refdes,
    date,
    fill_value,
    method,
    sr,
    wav_data_subtype,
    normalize_traces,
    verbose=False,
):
    hyd = HydrophoneDay(hyd_refdes, date)

    hyd.read_and_repair_gaps(fill_value=fill_value, method=method, wav_data_subtype=wav_data_subtype)

    # make dirs 
    date_str = datetime.strftime(hyd.date, "%Y_%m_%d")
    wav_dir = Path(f'./acoustic/wav/{date_str}')
    wav_dir.mkdir(parents=True, exist_ok=True)

    for st in tqdm(hyd.clean_list):
        start_time = str(st[0].stats['starttime'])
        dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    
        new_format = dt.strftime("%y%m%d%H%M%S%z")

        if wav_data_subtype == 'FLOAT':
            st[0].data = st[0].data.astype(np.float64) 
            
        if normalize_traces:
            st = st.normalize()
        
        wav_path = wav_dir / f"{new_format}.wav"

        if verbose:
            print(type(st[0].data[0]))
            print(str(wav_path))
    
        sf.write(wav_path, st[0].data, sr, subtype=wav_data_subtype) # use sf package to write instead of obspy

    return hyd