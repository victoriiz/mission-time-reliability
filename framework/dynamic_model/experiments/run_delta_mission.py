"""Reproduce the H100 lecture case without enumerating component states."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from delta_mission import DeltaMissionConfig, DeltaMissionModel, spare_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates', choices=('lecture', 'measured'), default='lecture')
    parser.add_argument('--nodes', type=int, default=160)
    parser.add_argument('--required-gpus', type=int, default=608)
    parser.add_argument('--steps', type=int, default=7200)
    parser.add_argument('--step-hours', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--spares', type=int, nargs='+', default=[0, 4, 6, 8, 10, 12])
    parser.add_argument('--paths', type=int, default=0, help='optional naive Monte Carlo paths')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, help='write summary and spare sweep as JSON')
    parser.add_argument('--curve', type=Path, help='write the cumulative mission curve as CSV')
    parser.add_argument('--sensitivity', action='store_true', help='also run uncalibrated gamma/eta scenarios')
    args = parser.parse_args()
    if args.paths < 0:
        parser.error('--paths must be nonnegative')
    if args.rates == 'lecture' and args.step_hours != 0.1:
        parser.error('rounded lecture probabilities require --step-hours 0.1; use --rates measured')
    kwargs = dict(nodes=args.nodes, required_gpus=args.required_gpus, steps=args.steps,
                  step_hours=args.step_hours, gamma=args.gamma, eta=args.eta)
    cfg = (DeltaMissionConfig(**kwargs) if args.rates == 'lecture'
           else DeltaMissionConfig.from_measurements(**kwargs))
    model = DeltaMissionModel(cfg)
    result = {'rates_mode': args.rates, 'baseline': model.summary(),
              'spare_sweep': spare_sweep(cfg, args.spares)}
    if args.sensitivity:
        result['sensitivity'] = [DeltaMissionModel(replace(cfg, gamma=g, eta=e)).summary()
                                 for g, e in [(0, 0), (0.006, 0), (0.06, 0), (0, 0.25), (0, 1)]]
    if args.paths:
        result['monte_carlo'] = model.simulate(args.paths, args.seed)
    if args.curve:
        args.curve.parent.mkdir(parents=True, exist_ok=True)
        with args.curve.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['step', 'hours', 'failure_probability'])
            writer.writerows((t, t*cfg.step_hours, p) for t, p in enumerate(model.mission_curve()))
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + '\n')


if __name__ == '__main__':
    main()
