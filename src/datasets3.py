import numpy as np
import soundfile as sf
from sklearn.preprocessing import normalize


def crop_or_pad(y, sr, period, record, mode="train"):
    len_y = len(y)
    effective_length = sr * period
    rint = np.random.randint(len(record['t_min']))
    time_start = record['t_min'][rint] * sr
    time_end = record['t_max'][rint] * sr
    if len_y > effective_length:
        # Positioning sound slice
        center = np.round((time_start + time_end) / 2)
        beginning = center - effective_length / 2
        if beginning < 0:
            beginning = 0
        beginning = np.random.randint(beginning, center)
        ending = beginning + effective_length
        if ending > len_y:
            ending = len_y
        beginning = ending - effective_length
        y = y[beginning:ending].astype(np.float32)
    else:
        y = y.astype(np.float32)
        beginning = 0
        ending = effective_length

    beginning_time = beginning / sr
    ending_time = ending / sr
    label = np.zeros(24, dtype='f')

    for i in range(len(record['t_min'])):
        if (record['t_min'][i] <= ending_time) and (record['t_max'][i] >= beginning_time):
            label[record['species_id'][i]] = 1

    return y, label


def crop_or_pad_v3(y, sr, period, record, mode="train"):
    len_y = len(y)
    effective_length = sr * period
    rint = np.random.randint(len(record['t_min']))
    time_start = record['t_min'][rint] * sr
    time_end = record['t_max'][rint] * sr
    if len_y > effective_length:
        # Positioning sound slice
        center = np.round((time_start + time_end) / 2)
        beginning = center - effective_length / 2
        if beginning < 0:
            beginning = 0
        beginning = np.random.randint(beginning, center)
        ending = beginning + effective_length
        if ending > len_y:
            ending = len_y
        beginning = ending - effective_length
        y = y[beginning:ending].astype(np.float32)
    else:
        y = y.astype(np.float32)
        beginning = 0
        ending = effective_length

    beginning_time = beginning / sr
    ending_time = ending / sr
    label = np.zeros(24, dtype='f')

    for i in range(len(record['t_min'])):
        if (record['t_min'][i] <= ending_time) and (record['t_max'][i] >= beginning_time):
            if record['is_add']:
                label[record['species_id'][i]] = 0.5
            else:
                label[record['species_id'][i]] = 1

    return y, label


class SedDataset:
    def __init__(self, df, period=10, stride=5, 
                 audio_transform=None, 
                 wave_form_mix_up_ratio=None,
                 data_path="train", mode="train"):

        self.period = period
        self.stride = stride
        self.audio_transform = audio_transform
        self.wave_form_mix_up_ratio = wave_form_mix_up_ratio
        self.data_path = data_path
        self.mode = mode

        self.df = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()
        self.len_df = len(self.df)

    def __len__(self):
        return self.len_df
    
    def __getitem__(self, idx):
        record = self.df.iloc[idx]

        y, sr = sf.read(f"{self.data_path}/{record['recording_id']}.flac")
        
        if self.mode != "test":
            y, label = crop_or_pad_v3(y, sr, period=self.period, record=record, mode=self.mode)

            if self.audio_transform:
                y = self.audio_transform(samples=y, sample_rate=sr)
                
            if self.wave_form_mix_up_ratio:
                if np.random.random() > 0.5:
                    # do mixup    
                    if idx < self.len_df//2:
                        rand_idx = np.random.randint(idx, self.len_df//2)
                    else:
                        rand_idx = np.random.randint(self.len_df//2, self.len_df)
                    record2 = self.df.iloc[rand_idx]
                    y2, sr2 = sf.read(f"{self.data_path}/{record2['recording_id']}.flac")
                    y2, label2 = crop_or_pad_v3(y2, sr2, period=self.period, record=record2, mode=self.mode)
                    
                    y = y * self.wave_form_mix_up_ratio + y2 * (1- self.wave_form_mix_up_ratio)
                    label = label * self.wave_form_mix_up_ratio + label2 * (1- self.wave_form_mix_up_ratio)
                else:
                    pass
        
        return {
            "image" : y,
            "target" : label,
            "id" : record['recording_id']
        }


