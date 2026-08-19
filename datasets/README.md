# Local Datasets

Place dataset files here using the names configured in
`experiments/config.yaml`.

Dataset files are ignored by git by default. Check the redistribution terms of
each source before publishing raw data.

## Paper-Level Numerical Datasets

The reported experiment uses eleven public numerical datasets:

| Config name | Expected local file | Source family |
| --- | --- | --- |
| `breast_cancer` | `breast_cancer.csv` | UCI |
| `ionosphere` | `ionosphere.data` | UCI |
| `parkinsons` | `parkinsons.csv` | UCI |
| `rice` | `rice_pr.csv` | UCI |
| `sonar` | `sonar.csv` | UCI |
| `spambase` | `spambase.csv` | UCI |
| `waveform` | `waveform.csv` | UCI |
| `vehicle` | `vehicle.csv` | UCI |
| `page_blocks` | `page_blocks.csv` | UCI |
| `glass` | `glass.data` | UCI |
| `image_segmentation` | `image_segmentation.csv` | UCI |

The runner expects target/drop-column metadata from `experiments/config.yaml`.

## Optional Local Dataset

`churn_pr.csv` is kept as an optional local dataset entry, but it is disabled in
the default config because the reported benchmark is restricted to numerical
datasets and because imported interval-pattern baselines are numeric-only.
