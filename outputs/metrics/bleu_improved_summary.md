| method | mean_bleu4 | std_bleu4 |
|---|---:|---:|
| epsilon | 0.1742 | 0.2486 |
| add_one | 0.2390 | 0.2059 |
| chen_cherry | 0.2366 | 0.2073 |
## Run: 2025-09-18T10:24:11.469811

| method | mean_bleu4 | std_bleu4 | good_range |
|---|---:|---:|---|
| epsilon | 0.1742 | 0.2486 | >=0.30 good; 0.20-0.30 moderate; <0.20 poor |
| add_one | 0.2390 | 0.2059 | >=0.30 good; 0.20-0.30 moderate; <0.20 poor |
| chen_cherry | 0.2366 | 0.2073 | >=0.30 good; 0.20-0.30 moderate; <0.20 poor |

## Run: 2025-09-18T10:33:22.013535

| method | mean_bleu4 | std_bleu4 | good_range | explanation |
|---|---:|---:|---|---|
| epsilon | 0.1742 | 0.2486 | >=0.30 good; 0.20-0.30 moderate; <0.20 poor | Low 4-gram overlap — short reports, paraphrasing, or many samples lack 4-gram matches. Smoothing applied in 60% of samples, indicating many missing higher-order n-grams. |
| add_one | 0.2390 | 0.2059 | >=0.30 good; 0.20-0.30 moderate; <0.20 poor | Moderate 4-gram overlap — some paraphrasing reduces exact 4-gram matches. Smoothing applied in 60% of samples, indicating many missing higher-order n-grams. |
| chen_cherry | 0.2366 | 0.2073 | >=0.30 good; 0.20-0.30 moderate; <0.20 poor | Moderate 4-gram overlap — some paraphrasing reduces exact 4-gram matches. Smoothing applied in 60% of samples, indicating many missing higher-order n-grams. |
