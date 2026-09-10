"""FROZEN COPY of models/metric at commit d979b1c (2026-06-03, "baseline finished").

This is the trainer that produced pretrained/baseline_rectangle_env1.pt, kept so
the June-era pipeline can be reproduced and compared against the current one.
Only the intra-package import lines and the source_folder path were rewritten;
everything else is verbatim.  Do not edit.

Differences from models/metric as it stands today, and from the upstream copy in
baselines/ntrl-demo/models/metric:

    Learning Rate        5e-5   (baselines/main: 1e-3)
    alpha                1.0    (baselines/main: 1.025)
    Save Every * Epoch   500    (baselines/main: 100)
    data_mlp             loads points / speed / normal only (today's also loads
                         speed_angles, speed_dists, trans_n, rot_n)

model_network_metric.py is byte-identical to the baselines copy, and
model_function_metric.py's Loss and TravelTimes are byte-identical to it too, so
checkpoints trained here can be evaluated with tests/3d_plan.py --models metric.
"""
