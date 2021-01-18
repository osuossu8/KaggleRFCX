import audiomentations as AA


train_audio_transform = AA.Compose([
    AA.AddGaussianNoise(p=0.5),
    AA.AddGaussianSNR(p=0.5),
    #AA.AddBackgroundNoise("../input/train_audio/", p=1)
    #AA.AddImpulseResponse(p=0.1),
    #AA.AddShortNoises("../input/train_audio/", p=1)
    #AA.FrequencyMask(min_frequency_band=0.0,  max_frequency_band=0.2, p=0.1),
    #AA.TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.1),
    #AA.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1),
    #AA.Shift(p=0.1),
    #AA.Normalize(p=0.1),
    #AA.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=1, p=0.05),
    #AA.PolarityInversion(p=0.05),
    #AA.Gain(p=0.2)
])
