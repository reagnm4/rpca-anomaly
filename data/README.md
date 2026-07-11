# data/

Not committed to git. After running `scripts/download_ucsd.sh`, layout is:

```
data/UCSD_Anomaly_Dataset.v1p2/
  UCSDped2/
    Train/Train001 ... Train016/   # normal-only clips, ~120-180 .tif frames each
    Test/Test001  ... Test012/     # clips containing anomalies
    Test/Test001_gt ... /          # pixel-level ground-truth masks (subset)
    UCSDped2.m                     # frame-level ground-truth ranges per test clip
```

Frames are individual grayscale TIFFs, 240x360, named 001.tif, 002.tif, ...
There are no video files — "frame extraction" here means loading + ordering
the TIFF sequence, not decoding video.
