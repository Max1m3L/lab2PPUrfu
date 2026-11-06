"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations
import pandas as pd
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary

def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.file_path)

    missing_counts = df.isna().sum().to_dict()

    passengers_above_30 = int((df['Age'] > 30).sum())

    pclass_avg_age = df.groupby('Pclass')['Age'].mean().to_dict()

    pclass_survival_rates = df.groupby('Pclass')['Survived'].mean().to_dict()

    top_5_fares = df['Fare'].sort_values(ascending=False).head(5).tolist()

    pclass_avg_age = {int(k): float(v) for k, v in pclass_avg_age.items()}
    pclass_survival_rates = {int(k): float(v) for k, v in pclass_survival_rates.items()}

    return TitanicSummary(
        missing_counts=missing_counts,
        passengers_above_30=passengers_above_30,
        pclass_avg_age=pclass_avg_age,
        pclass_survival_rates=pclass_survival_rates,
        top_5_fares=top_5_fares
    )
