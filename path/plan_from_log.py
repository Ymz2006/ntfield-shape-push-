"""Plan a reference path to the goal from every shape pose in a run log.

    python path/plan_from_log.py --shape-name T    path/far/log.txt    path/far
    python path/plan_from_log.py --shape-name rect path/rectangle/raw  path/rectangle
    python path/plan_from_log.py --shape-name V    path/V/raw          path/V

Reads the ``[run] #NNN  T=(x, y, deg)`` lines, and writes into OUT:

* ``actual.npy``      (N,6) the observed poses in the planner's normalized frame
* ``actual_cm.npy``   (N,3) the same poses in table cm + radians
* ``traj.npy``        (N,T,6) the MPPI plan from pose i to the goal, normalized, in
                      the same (x, y, z, rx, ry, rz) layout as ``actual.npy``.  Plans
                      are different lengths, so each is padded to T by repeating its
                      last waypoint; ``traj_len.npy`` (N,) holds the real lengths
* ``traj_cm/NNN.npy`` (T_i,3) plan NNN resampled the way the controller tracks it,
                      in table cm + radians
* ``summary.txt``
"""
import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shapes  # noqa: E402

shapes.select_from_argv()

import push_t_demo_realworld as rw  # noqa: E402
from push_t_demo_sim import plan_from, planner_to_world, resample_path  # noqa: E402

POSE_RE = re.compile(r'\[run\]\s+#(\d+)\s+T=\(\s*([-\d.]+),\s*([-\d.]+),\s*([-+\d.]+)\)')


def parse_log(path):
    poses = []
    with open(path) as fh:
        for line in fh:
            m = POSE_RE.search(line)
            if m:
                poses.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                              math.radians(float(m.group(4)))))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('out')
    shapes.add_shape_argument(ap)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    cli = ap.parse_args()

    import torch
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    args = rw.make_args(plan_device=cli.device)
    geo = rw.load_geometry(args)
    womodel = rw.load_womodel(args)
    goal_norm = rw.goal_pose_norm()
    goal_cm = rw.planner_pose_to_real(goal_norm, geo.env_scale, geo.env_center,
                                      geo.bbox_to_centroid)
    print(f'[plan] shape {shapes.active().name}: goal norm '
          f'({goal_norm[0]:+.3f}, {goal_norm[1]:+.3f}, {goal_norm[5]:+.3f} turns) == '
          f'({goal_cm[0]:.1f}, {goal_cm[1]:.1f}, {math.degrees(goal_cm[2]):+.0f}deg) cm')

    poses = parse_log(cli.log)
    print(f'[plan] {len(poses)} poses in {cli.log}')

    os.makedirs(os.path.join(cli.out, 'traj_cm'), exist_ok=True)

    actual_norm, actual_cm, trajs, lines = [], [], [], []
    for idx, x, y, th in poses:
        start_norm = rw.real_pose_to_planner((x, y, th), geo.env_scale, geo.env_center,
                                             geo.bbox_to_centroid)
        path_norm, dist = plan_from(womodel, start_norm, goal_norm, cli.device,
                                    args.mppi_steps)
        path_norm = np.asarray(path_norm, dtype=float)
        ref_world = resample_path(planner_to_world(path_norm, geo.env_scale, geo.env_center,
                                                   geo.bbox_to_centroid),
                                  args.spacing, args.smooth)
        ref_cm = rw.path_sim_to_real(ref_world)

        traj6 = np.zeros((len(path_norm), 6))
        traj6[:, [0, 1, 5]] = path_norm

        actual_norm.append(start_norm)
        actual_cm.append([x, y, th])
        trajs.append(traj6)
        np.save(os.path.join(cli.out, 'traj_cm', f'{idx:03d}.npy'), ref_cm)

        dgoal = float(np.hypot(x - goal_cm[0], y - goal_cm[1]))
        line = (f'#{idx:03d}  T=({x:6.1f},{y:6.1f},{math.degrees(th):+4.0f})  '
                f'norm ({start_norm[0]:+.3f},{start_norm[1]:+.3f},{start_norm[5]:+.3f})  '
                f'|goal|={dgoal:5.1f}cm  plan {len(path_norm)} pts -> {len(ref_cm)} ref '
                f'waypoints, final |goal-x|={dist:.4f}')
        print('[plan] ' + line)
        lines.append(line)

    lens = np.array([len(t) for t in trajs])
    padded = np.stack([np.concatenate([t, np.repeat(t[-1:], lens.max() - len(t), axis=0)])
                       for t in trajs])
    np.save(os.path.join(cli.out, 'actual.npy'), np.array(actual_norm))
    np.save(os.path.join(cli.out, 'actual_cm.npy'), np.array(actual_cm))
    np.save(os.path.join(cli.out, 'traj.npy'), padded)
    np.save(os.path.join(cli.out, 'traj_len.npy'), lens)
    with open(os.path.join(cli.out, 'summary.txt'), 'w') as fh:
        fh.write(f'shape {shapes.active().name}  log {cli.log}\n')
        fh.write(f'goal norm ({goal_norm[0]:+.3f}, {goal_norm[1]:+.3f}, {goal_norm[5]:+.3f}) '
                 f'== ({goal_cm[0]:.1f}, {goal_cm[1]:.1f}, {math.degrees(goal_cm[2]):+.0f}deg) cm\n')
        fh.write('\n'.join(lines) + '\n')
    print(f'[plan] wrote traj.npy {padded.shape} (lengths {lens.min()}..{lens.max()}) '
          f'to {cli.out}')


if __name__ == '__main__':
    main()
