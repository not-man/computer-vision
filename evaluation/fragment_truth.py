"""Ручная разметка толпы 17:45:25-17:45:47 по одежде на раскадровках: камера, номер трекера и время -> человек.

Привязка к выходу детектора, а не к номерам кусков: номера кусков меняются при любой правке склейки.
"""
import pandas as pd
from scipy.optimize import linear_sum_assignment

WINDOW = (2725.0, 2747.0)
LABELS = [
    ('camera_new_3', 415, 2725.06, 2726.06, 'принт'),
    ('camera_new_3', 419, 2725.06, 2725.46, 'лого'),
    ('camera_new_3', 425, 2732.46, 2734.06, 'белая'),
    ('camera_new_3', 426, 2732.98, 2739.98, 'полосатая'),
    ('camera_new_3', 428, 2733.38, 2738.98, 'толстовка'),
    ('camera_new_3', 428, 2740.06, 2746.98, 'куртка'),
    ('camera_new_3', 429, 2733.46, 2739.18, 'юбка'),
    ('camera_new_3', 437, 2735.78, 2739.98, 'белая'),
    ('camera_new_3', 437, 2740.06, 2741.98, 'толстовка'),
    ('camera_new_3', 459, 2739.98, 2742.26, 'юбка'),
    ('camera_new_3', 476, 2741.18, 2742.86, 'принт'),
    ('camera_new_3', 482, 2741.98, 2746.98, 'лого'),
    ('camera_new_3', 488, 2742.66, 2746.98, 'принт'),
    ('camera_new_4', 142, 2725.67, 2731.77, 'лого'),
    ('camera_new_4', 146, 2725.07, 2730.67, 'принт'),
    ('camera_new_4', 148, 2725.07, 2727.27, 'полосатая'),
    ('camera_new_4', 151, 2727.57, 2736.27, 'полосатая'),
    ('camera_new_4', 153, 2728.77, 2735.77, 'юбка'),
    ('camera_new_4', 158, 2735.67, 2742.67, 'принт'),
    ('camera_new_4', 161, 2736.57, 2741.77, 'лого'),
    ('camera_new_4', 162, 2738.07, 2745.17, 'белая'),
    ('camera_new_4', 163, 2739.07, 2746.97, 'толстовка'),
    ('camera_new_4', 165, 2739.87, 2743.17, 'куртка'),
    ('camera_new_4', 177, 2745.67, 2745.77, 'белая'),
    ('camera_new_4', 178, 2745.77, 2746.37, 'белая'),
    ('camera_new_5', 269, 2725.07, 2725.67, 'принт'),
    ('camera_new_5', 280, 2725.07, 2725.27, 'лого'),
    ('camera_new_5', 283, 2731.67, 2736.75, 'белая'),
    ('camera_new_5', 284, 2732.47, 2734.95, 'толстовка'),
    ('camera_new_5', 289, 2733.67, 2746.95, 'юбка'),
    ('camera_new_5', 291, 2733.87, 2737.55, 'полосатая'),
    ('camera_new_5', 292, 2733.95, 2741.95, 'куртка'),
    ('camera_new_5', 292, 2742.07, 2746.95, 'принт'),
    ('camera_new_5', 294, 2735.27, 2737.35, 'толстовка'),
    ('camera_new_5', 306, 2737.55, 2737.95, 'белая'),
    ('camera_new_5', 313, 2740.07, 2746.95, 'лого'),
    ('camera_new_5', 326, 2742.35, 2746.95, 'полосатая'),
    ('camera_new_5', 328, 2742.87, 2743.75, 'куртка'),
]
TIME_TOLERANCE_S = 0.02  # границы записаны с округлением

# Проверка глазами всех треков записи 7 ноября. Точка: камера, номер трекера, секунда по общим часам.
SAME_PERSON = [
    (('camera_new_3', 716, 3101.4), ('camera_new_3', 718, 3123.5), 'женщина в розовых брюках'),
    (('camera_new_3', 581, 2914.7), ('camera_new_4', 214, 3024.5), 'женщина в коричневом'),
    (('camera_new_5', 242, 2704.6), ('camera_new_5', 326, 2745.3), 'женщина в полосатом'),
    (('camera_new_3', 721, 3111.8), ('camera_new_3', 751, 3218.6), 'мужчина в синем пиджаке'),
    (('camera_new_3', 738, 3186.1), ('camera_new_4', 265, 3209.7), 'мужчина в чёрной футболке'),
    (('camera_new_5', 136, 2528.2), ('camera_new_3', 149, 2531.2), 'мужчина в чёрной футболке с рисунком'),
    (('camera_new_3', 412, 2714.1), ('camera_new_3', 322, 2648.2), 'человек в чёрном мешковатом'),
    (('camera_new_3', 84, 2479.9), ('camera_new_3', 149, 2531.2), 'мужчина в чёрной футболке рядом с белой'),
    (('camera_new_3', 17, 2345.8), ('camera_new_5', 5, 2349.4), 'седой мужчина в чёрном свитере'),
    (('camera_new_4', 158, 2739.2), ('camera_new_5', 269, 2720.3), 'мужчина в чёрной футболке и джинсах'),
    (('camera_new_5', 162, 2574.5), ('camera_new_3', 205, 2572.7), 'мужчина в серых брюках'),
    (('camera_new_3', 215, 2583.7), ('camera_new_3', 205, 2572.7), 'мужчина в серых брюках'),
    (('camera_new_4', 235, 3148.1), ('camera_new_4', 218, 3042.4), 'мужчина в синем и бежевых брюках'),
    (('camera_new_5', 203, 2632.2), ('camera_new_3', 147, 2528.7), 'мужчина в чёрном у камеры'),
]
DIFFERENT_PEOPLE = [
    (('camera_new_5', 148, 2558.1), ('camera_new_3', 235, 2620.8), 'мужчина в белом и женщина в юбке'),
    (('camera_new_3', 869, 3357.8), ('camera_new_3', 897, 3391.1), 'седой в джинсах и мужчина в чёрных брюках'),
    (('camera_new_5', 194, 2627.2), ('camera_new_3', 297, 2644.9), 'женщина с хвостом и мужчина в белом'),
    (('camera_new_4', 240, 3138.7), ('camera_new_4', 214, 3024.5), 'мужчина в сиреневом и женщина в коричневом'),
    (('camera_new_4', 218, 3042.4), ('camera_new_4', 214, 3024.5), 'пара в обнимку'),
    (('camera_new_3', 718, 3123.5), ('camera_new_4', 240, 3138.7), 'женщина в розовых брюках и мужчина в сиреневом'),
    (('camera_new_4', 235, 3148.1), ('camera_new_3', 718, 3123.5), 'мужчина в синем и женщина в розовых брюках'),
    (('camera_new_3', 581, 2914.7), ('camera_new_4', 218, 3042.4), 'пара в обнимку'),
    (('camera_new_3', 1, 2326.0), ('camera_new_3', 17, 2345.8), 'женщина и седой мужчина'),
    (('camera_new_3', 147, 2528.7), ('camera_new_3', 205, 2572.7), 'чёрное худи и серые брюки'),
    (('camera_new_5', 148, 2558.1), ('camera_new_3', 147, 2528.7), 'белая футболка и чёрное худи'),
    (('camera_new_3', 176, 2555.6), ('camera_new_3', 149, 2531.2), 'чёрное мешковатое и чёрная футболка'),
    (('camera_new_5', 26, 2426.7), ('camera_new_3', 149, 2531.2), 'двое в чёрных футболках'),
    (('camera_new_3', 297, 2644.9), ('camera_new_5', 148, 2558.1), 'два прихода мужчины в белом, пауза больше минуты'),
]


def label_rows(rows: pd.DataFrame) -> pd.Series:
    """Человек для каждой строки наблюдений или None, если строка не размечена."""
    people = pd.Series(None, index=rows.index, dtype=object)
    for camera, raw_track, start, end, person in LABELS:
        chosen = (rows.camera == camera) & (rows.raw_track == raw_track) & (rows.t_s >= start - TIME_TOLERANCE_S) & (rows.t_s <= end + TIME_TOLERANCE_S)
        people[chosen] = person
    return people


def score(rows: pd.DataFrame, track_of_row: pd.Series, people: pd.Series | None = None) -> dict:
    """Соответствие человек-трек в окне. rows: наблюдения с camera, raw_track, t_s; track_of_row: итоговый номер трека для тех же строк.

    people: кто в какой строке, если разметка перенесена на другой прогон детектора.
    """
    table = pd.DataFrame({"person": label_rows(rows) if people is None else people, "track": track_of_row.reindex(rows.index)}).dropna()
    crossed = pd.crosstab(table.person, table.track)
    total = int(crossed.values.sum())
    # одно лучшее соответствие человек-трек: как IDF1 в этом окне
    r, c = linear_sum_assignment(-crossed.values)
    matched = int(crossed.values[r, c].sum())
    return dict(строк=total, соответствие=round(matched / total, 3), чистота=round(float(crossed.max(axis=0).sum() / total), 3),
                полнота=round(float(crossed.max(axis=1).sum() / total), 3), людей=int(crossed.shape[0]), треков=int(crossed.shape[1]),
                по_людям={p: {int(k): int(v) for k, v in crossed.loc[p][crossed.loc[p] > 0].sort_values(ascending=False).items()} for p in crossed.index})


def score_fixed_truth(people: pd.Series, track_of_row: pd.Series) -> dict:
    """Сопоставление ID на фиксированном числе размеченных исходных рамок.

    Отсутствие новой детекции или итогового трека остаётся в знаменателе. Чистота считается
    только по назначенным рамкам и потому отдельно от соответствия показывает смешение ID.
    """
    if people.isna().any() or not people.index.equals(track_of_row.index):
        raise ValueError("Нужны размеченные строки и выровненные по индексу предсказания")
    table = pd.DataFrame({"person": people, "track": track_of_row}).dropna()
    total, covered = len(people), len(table)
    if not total:
        raise ValueError("Нет размеченных строк")
    if table.empty:
        return {"размечено": total, "в_треках": 0, "совпало": 0, "соответствие": 0.0, "чистота": 0.0}
    crossed = pd.crosstab(table.person, table.track)
    rows, columns = linear_sum_assignment(-crossed.values)
    matched = int(crossed.values[rows, columns].sum())
    return {"размечено": total, "в_треках": covered, "совпало": matched,
            "соответствие": round(matched / total, 3),
            "чистота": round(float(crossed.max(axis=0).sum() / covered), 3)}


def track_on_truth(truth_rows: pd.DataFrame, assigned: pd.DataFrame, track_of_row: pd.Series, *, target: str) -> pd.Series:
    """Итоговый ID для каждой исходной размеченной рамки; пропуск остаётся пустым."""
    keys = ["camera", "frame", "raw_track"]
    if assigned.duplicated(keys).any():
        raise ValueError("Один номер трекера повторяется в кадре среди наблюдений")
    if not assigned.index.equals(track_of_row.index):
        raise ValueError("Индексы наблюдений и итоговых номеров не совпадают")
    reference = assigned[keys].copy()
    reference["track"] = track_of_row.to_numpy()
    labels = truth_rows[["camera", "frame", target]].rename(columns={target: "raw_track"}).copy()
    labels["raw_track"] = labels.raw_track.astype("Int64")
    joined = labels.merge(reference, on=keys, how="left", validate="many_to_one", sort=False)
    return pd.Series(joined.track.to_numpy(), index=truth_rows.index)


def check_pairs(rows: pd.DataFrame, track_of_row: pd.Series) -> dict[str, list[str]]:
    """Сверка с проверкой глазами: кто должен быть в одном треке и кто в разных. Точка ищет ближайшую строку своего номера трекера."""
    def track_at(point: tuple) -> float | None:
        camera, raw_track, t = point
        own = rows[(rows.camera == camera) & (rows.raw_track == raw_track)]
        own = own[(own.t_s - t).abs() <= 2.0]
        if own.empty:
            return None
        nearest = track_of_row.reindex(own.index).dropna()
        if nearest.empty:
            return None
        return float(nearest.iloc[int((own.loc[nearest.index, "t_s"] - t).abs().to_numpy().argmin())])

    apart = [text for a, b, text in SAME_PERSON if track_at(a) is None or track_at(a) != track_at(b)]
    together = [text for a, b, text in DIFFERENT_PEOPLE if track_at(a) is not None and track_at(a) == track_at(b)]
    return {"должны быть вместе, но врозь": apart, "должны быть врозь, но вместе": together}
