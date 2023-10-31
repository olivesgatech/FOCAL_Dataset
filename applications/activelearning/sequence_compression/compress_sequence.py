import spacial_temporal_compression
import numpy as np
import os

seq_path = "../../../data/deepenneo_pcd_customdata-s6nwj6hi-defaultproject-gCIjaypSK93PmLOtC1wtmScJ/processed/images"

# for each consecutive frame in the sequence run the 3-Step Search Block Matching Algorithm
all_frames = sorted(os.listdir(seq_path))
all_residualMetrics = []
all_naiveResidualMetrics = []

for i in range(len(os.listdir(seq_path)) - 1):
    anchorFrame = os.path.join(seq_path, all_frames[i])
    targetFrame = os.path.join(seq_path, all_frames[i + 1])
    residualMetric, naiveResidualMetric = spacial_temporal_compression.main(
                                                                      anchorFrame,
                                                                      targetFrame,
                                                                      outfile='OUTPUT' + str(i),
                                                                      saveOutput=True)
    all_residualMetrics.append(residualMetric)
    all_naiveResidualMetrics.append(naiveResidualMetric)

metricRange = np.arange(len(all_residualMetrics))
spacial_temporal_compression.plot(metricRange, all_residualMetrics,
                                  'deepenneo_pcd_customdata-s6nwj6hi-defaultproject-gCIjaypSK93PmLOtC1wtmScJ',
                                  ylabel='residualMetric')
spacial_temporal_compression.plot(metricRange, all_naiveResidualMetrics,
                                  'deepenneo_pcd_customdata-s6nwj6hi-defaultproject-gCIjaypSK93PmLOtC1wtmScJ',
                                  ylabel='naiveResidualMetric')

