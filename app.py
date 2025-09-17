def compute_group_stats(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_col, dropna=True)[metric]
    for treatment, values in grouped:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) == 0:
            continue
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        # ✅ Fixed handling of whiskers (no 'initial' keyword)
        vals_in_lower = values[values >= (q1 - 1.5 * iqr)]
        vals_in_upper = values[values <= (q3 + 1.5 * iqr)]
        lower_whisker = vals_in_lower.min() if not vals_in_lower.empty else np.nan
        upper_whisker = vals_in_upper.max() if not vals_in_upper.empty else np.nan

        outliers = values[(values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr))]

        rows.append({
            "Treatment": treatment,
            "N": int(values.size),
            "Mean": float(values.mean()),
            "Median": float(values.median()),
            "Q1": float(q1),
            "Q3": float(q3),
            "Std Dev": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "Min": float(values.min()),
            "Max": float(values.max()),
            "Lower Whisker": float(lower_whisker) if not np.isnan(lower_whisker) else None,
            "Upper Whisker": float(upper_whisker) if not np.isnan(upper_whisker) else None,
            "Outliers": int(outliers.size),
        })
    return pd.DataFrame(rows)
