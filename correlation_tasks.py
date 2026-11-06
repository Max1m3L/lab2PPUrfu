"""Задачи второй части лабораторной: Корреляционный анализ."""
from __future__ import annotations
import pandas as pd
from grader_contracts.correlation_tasks import BrainDataInput, BrainCorrelationSummary


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Выполните корреляционный анализ размера мозга.

    Нужно загрузить данные с разделителем tab, обработать пропуски NA,
    разделить на мужчин/женщин и посчитать корреляции Пирсона с MRI_Count.
    """
    df = pd.read_csv(data.file_path, sep='\t', na_values='NA')

    men_df = df[df['Gender'] == 'Male']
    women_df = df[df['Gender'] == 'Female']

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    male_correlations = men_df[features].corrwith(men_df['MRI_Count'], method='pearson').to_dict()

    female_correlations = women_df[features].corrwith(women_df['MRI_Count'], method='pearson').to_dict()

    max_corr_val = -1.0
    strongest_mri_feature = ""

    for feat in features:
        if feat in male_correlations and pd.notna(male_correlations[feat]):
            if abs(male_correlations[feat]) > max_corr_val:
                max_corr_val = abs(male_correlations[feat])
                strongest_mri_feature = feat

        if feat in female_correlations and pd.notna(female_correlations[feat]):
            if abs(female_correlations[feat]) > max_corr_val:
                max_corr_val = abs(female_correlations[feat])
                strongest_mri_feature = feat

    return BrainCorrelationSummary(
        male_correlations=male_correlations,
        female_correlations=female_correlations,
        strongest_mri_feature=strongest_mri_feature
    )
