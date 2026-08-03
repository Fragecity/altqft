# Model Registry

This table maps saved parameter files to their trained model configuration.

| Parameter file | Checkpoint | Model | Objective | nqubit | Periods | alpha | beta | Shift loss | Shift samples | Best epoch | Best min FI | Best loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ph1_hp1_shared_fi_shift_10q_p2-99_phases.json | ph1_hp1_shared_fi_shift_10q_p2-99.pt | HP1_shared | hp1_shared_fi_shift | 10 | 2-99 | 1 | 3.8702218 | kl_to_shift_mean | 8x8 | 120 | 14.254809 | -5.0293083 |
| ph1_hp1_shared_fi_shift_11q_p2-120_phases.json | ph1_hp1_shared_fi_shift_11q_p2-120.pt | HP1_shared | hp1_shared_fi_shift | 11 | 2-120 | 1 | 1.6109601 | kl_to_shift_mean | 8x8 | 285 | 19.534958 | -12.498734 |
| ph1_hp1_shared_fi_shift_18q_p2-511_phases.json | ph1_hp1_shared_fi_shift_18q_p2-511.pt | HP1_shared | hp1_shared_fi_shift | 18 | 2-511 | 1 | 14.374258 | kl_to_shift_mean | 8x8 | 142 | 111.05581 | -53.771091 |
| ph1_hp1_shared_fi_shift_8q_p2-63_phases.json | ph1_hp1_shared_fi_shift_8q_p2-63.pt | HP1_shared | hp1_shared_fi_shift | 8 | 2-63 | 1 | 1.0736894 | kl_to_shift_mean | 8x8 | 174 | 6.7048159 | -6.1811104 |
| ph1_hp1_shared_fi_shift_9q_p2-80_phases.json | ph1_hp1_shared_fi_shift_9q_p2-80.pt | HP1_shared | hp1_shared_fi_shift | 9 | 2-80 | 1 | 0.66434511 | kl_to_shift_mean | 8x8 | 390 | 12.544685 | -9.4914703 |

## Period decoder

| Checkpoint | nqubit | Periods | Decoder | Beam width | Selected epoch | Validation top-1/top-4 |
| --- | --- | --- | --- | --- | --- | --- |
| period_recovery_distribution_18q_p2-511_nibble_ddp/selected.pt | 18 | 2-511 | 4-bit autoregressive | 4 | 390 | 1.0000 / 1.0000 |
