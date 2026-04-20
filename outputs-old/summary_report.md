# Summary Report

## Strongest Between-Model Difference
- Prompt: **up_cot**
- Minimum ANOVA p-value across dimensions: **0.001324**
- Maximum ANOVA F-statistic across dimensions: **64.000**

## Ranking (by min ANOVA p-value)

| prompt_name   |   anova_f_physical |   anova_p_physical |   anova_f_cognitive |   anova_p_cognitive |   min_anova_p |   max_anova_f | significant_any_dim   |
|:--------------|-------------------:|-------------------:|--------------------:|--------------------:|--------------:|--------------:|:----------------------|
| up_cot        |                1   |           0.373901 |                  64 |           0.0013239 |     0.0013239 |          64   | True                  |
| base_cot      |                2.7 |           0.175693 |                   0 |           1         |     0.175693  |           2.7 | False                 |

## Group Comparison Plots
- ![anova_groups_base_cot.png](anova_groups_base_cot.png)
- ![anova_groups_up_cot.png](anova_groups_up_cot.png)

## Significance Flags
- up_cot shows significant between-model differences (min p=0.001324)
