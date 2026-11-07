"""Ручная разметка толпы 17:45:25-17:45:47 по одежде на раскадровках: кусок и время -> человек."""
import numpy as np
import pandas as pd

WINDOW = (2725.0, 2747.0)
SIMPLE = {
    235: "полосатая", 236: "полосатая", 379: "полосатая", 386: "полосатая",
    237: "юбка", 133: "юбка", 378: "юбка",
    239: "принт", 138: "принт", 140: "принт",
    240: "лого", 139: "лого",
    241: "белая", 244: "белая", 245: "белая", 130: "белая", 375: "белая", 382: "белая",
    242: "толстовка", 381: "толстовка",
    243: "куртка", 387: "куртка",
    232: "лого", 129: "лого", 373: "лого",  # на снимках тот же человек, что ушёл и вернулся через 5 с
    234: "принт", 128: "принт", 371: "принт",  # у к128 тот же принт на футболке, что у к239
}
TIMED = {  # (кусок: [(до какого времени, человек)]), None значит не размечено
    131: [(2740.0, "полосатая"), (9e9, None)],
    132: [(2739.0, "толстовка"), (2740.0, None), (9e9, "куртка")],
    134: [(2740.0, "белая"), (9e9, "толстовка")],
    135: [(2742.5, "юбка"), (9e9, None)],
    376: [(2735.5, "толстовка"), (9e9, None)],
    380: [(2742.0, "куртка"), (9e9, "принт")],
    384: [(2740.0, None), (9e9, "лого")],
}


def label(tracklet: int, t: float):
    if tracklet in SIMPLE:
        return SIMPLE[tracklet]
    for end, person in TIMED.get(tracklet, []):
        if t < end:
            return person
    return None


def score(base_assigned: pd.DataFrame, track_of_row: pd.Series) -> dict:
    """Base_assigned: исходные куски; track_of_row: итоговый номер трека для тех же строк."""
    rows = base_assigned[(base_assigned.t_s >= WINDOW[0]) & (base_assigned.t_s <= WINDOW[1])].copy()
    rows["person"] = [label(int(a), float(b)) for a, b in zip(rows.tracklet, rows.t_s)]
    rows["track"] = track_of_row.reindex(rows.index)
    rows = rows[rows.person.notna() & rows.track.notna()]
    table = pd.crosstab(rows.person, rows.track)
    total = int(table.values.sum())
    # одно лучшее соответствие человек-трек: как IDF1 в этом окне
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-table.values)
    matched = int(table.values[r, c].sum())
    return dict(строк=total, соответствие=round(matched / total, 3), чистота=round(float(table.max(axis=0).sum() / total), 3), полнота=round(float(table.max(axis=1).sum() / total), 3),
                людей=int(table.shape[0]), треков=int(table.shape[1]),
                по_людям={p: {int(k): int(v) for k, v in table.loc[p][table.loc[p] > 0].sort_values(ascending=False).items()} for p in table.index})
