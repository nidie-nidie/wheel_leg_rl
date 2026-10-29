from pathlib import Path

import numpy as np

from debug.sim2sim.action_mode_engine_probe_20261020 import PROJECT, SIM
from debug.sim2sim.audit_reward_economics_20261019 import digest, read_json, write_json, frozen_identity
from debug.sim2sim.verify_wheel_feedback_paths_20261020 import rows_of
from debug.sim2sim.wheel_feedback_paths_20261020 import OUTPUT


def main():
    before = frozen_identity()
    old_directory = PROJECT / 'artifacts/phase2_dreamwaq/baseline-heading-implementation-20261016/old701-mujoco-final'
    formal_directory = PROJECT / 'logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/evaluation'
    old = read_json(old_directory / 'summary.json')
    formal = read_json(formal_directory / 'summary.json')
    for filename, expected in formal['evaluation_contract']['runtime_source_sha256'].items():
        assert digest(SIM / 'wheelleg_mujoco' / filename) == expected
    assert old['evaluation_contract']['scenarios'] == formal['evaluation_contract']['scenarios']
    physical_fields = ('vx_mps', 'yaw_rate_rad_s', 'base_height_m', 'roll_rad', 'pitch_rad', 'yaw_rad',
                       'phi0_left_rad', 'phi0_right_rad', 'max_loop_closure_error_m', 'effort_saturation_fraction',
                       'target_vx_mps', 'target_yaw_rate_rad_s', 'target_base_height_m')
    physical_fields += tuple(f'raw_action_{index}' for index in range(6))
    physical_fields += tuple(f'leg_velocity_{index}_rad_s' for index in range(4))
    physical_fields += tuple(f'leg_acceleration_{index}_rad_s2' for index in range(4))
    results = []
    for record in old['groups']['disabled']['scenarios']:
        name = record['name']
        old_path = old_directory / 'disabled' / f'{name}.csv'
        new_path = formal_directory / 'disabled' / f'{name}.csv'
        assert digest(old_path) == record['csv_sha256']
        formal_record = next(row for row in formal['groups']['disabled']['scenarios'] if row['name'] == name)
        assert digest(new_path) == formal_record['csv_sha256']
        left, right = rows_of(old_path), rows_of(new_path)
        assert len(left) == len(right) == 500
        errors = {}
        for field in physical_fields:
            original = np.array([float(row[field]) for row in left])
            reproduced = np.array([float(row[field]) for row in right])
            errors[field] = float(np.max(np.abs(original - reproduced)))
            assert np.array_equal(original, reproduced), (name, field, errors[field])
        results.append({'name': name, 'old_csv_sha256': digest(old_path), 'current_source_formal_csv_sha256': digest(new_path),
                        'physical_fields_max_abs_error': errors})
    assert len(results) == 8 and frozen_identity() == before
    result = {'old_sha256': old['evaluation_contract']['runtime_source_sha256']['evaluation.py'],
              'current_sha256': digest(SIM / 'wheelleg_mujoco/evaluation.py'),
              'historical_source_text_fully_recovered': False, 'metadata_only_difference_asserted': False,
              'physical_trajectory_equal_all_eight': True, 'fields_per_trajectory': len(physical_fields),
              'method': 'Reused already completed formal4096/701 evaluation produced with current runtime sources. All listed physical/action/command traces match the legacy701 CSV numerically bit for bit. Legacy source text is not fully recovered, so source equivalence is not asserted.',
              'scenes': results, 'frozen_files_checked': 118, 'source_sha256': digest(__file__)}
    write_json(OUTPUT / 'old701_evaluation_source_audit.json', result)
    print({'physical_trajectories_exact': len(results), 'fields_per_trajectory': len(physical_fields)}, flush=True)


if __name__ == '__main__':
    main()
