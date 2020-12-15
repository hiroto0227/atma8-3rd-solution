import re
import unicodedata
from typing import Dict, List
from collections import Counter
from itertools import combinations
import json

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from pandas.core import series
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

import config

# import nltk
# nltk.download("stopwords")


def run(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[pd.DataFrame]:
    # 型やmissing valueを直す。
    train_df["User_Score"] = (
        train_df["User_Score"].replace("tbd", np.nan).astype(np.float32)
    )
    test_df["User_Score"] = (
        test_df["User_Score"].replace("tbd", np.nan).astype(np.float32)
    )

    # ============================================
    # === 破壊的変更
    # ============================================
    normalize_strings(train_df, test_df)
    combination_cols(train_df, test_df)
    rating_mapping(train_df, test_df)
    add_serirese_name(train_df, test_df)

    # feature data
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()

    preprocess_functions = [
        add_features,
        name_processing,  ## before groupby features
        groupby_features,
        groupby_target_features,
        lda_processing,
        rank_by_cols,
        label_encoding,  ## after groupby features
    ]  # List[Callable[[pd.DataFrame, pd.DataFrame], List[pd.DataFrame]]]

    for process in preprocess_functions:
        print(f"start process: {process.__name__}")
        _train_df, _test_df = process(train_df, test_df)

        train_features = pd.concat([train_features, _train_df], axis=1)
        test_features = pd.concat([test_features, _test_df], axis=1)

        assert len(train_features) == len(train_df)
        assert len(test_features) == len(test_df)

        print(len(train_features.columns))

    drop_cols: List[str] = []
    train_features.drop(columns=drop_cols, inplace=True)
    test_features.drop(columns=drop_cols, inplace=True)
    return [train_features, test_features]


def normalize_strings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    def get_first_normalize_string(x) -> str:
        return " ".join(
            re.split(
                r"[ ,]+",
                unicodedata.normalize(
                    "NFKC", re.sub(r"[:/(/)/+/-/&*-]", "", str(x).lower())
                ),
            )[:1]
        )

    train_df["norm_Developer"] = train_df["Developer"].apply(get_first_normalize_string)
    test_df["norm_Developer"] = test_df["Developer"].apply(get_first_normalize_string)
    train_df["norm_Publisher"] = train_df["Publisher"].apply(get_first_normalize_string)
    test_df["norm_Publisher"] = test_df["Publisher"].apply(get_first_normalize_string)

    def get_year_bins(year_of_release):
        if year_of_release < 1995:
            return 1995
        elif year_of_release < 2000:
            return 2000
        elif year_of_release < 2005:
            return 2005
        elif year_of_release < 2010:
            return 2010
        elif year_of_release < 2015:
            return 2015
        return 2020

    train_df["norm_Year_of_Release"] = train_df["Year_of_Release"].apply(get_year_bins)
    test_df["norm_Year_of_Release"] = test_df["Year_of_Release"].apply(get_year_bins)


def add_serirese_name(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    def _normalize_name(name: str) -> str:
        name = str(name)
        name = re.sub(r"-", " ", name)
        # name = re.sub(r"\d+", "0", name)
        return unicodedata.normalize(
            "NFKC", re.sub(r"[:/(/)/+/-/&*-,;!]", "", str(name).lower())
        )

    train_df["normalized_name"] = train_df.Name.apply(_normalize_name)
    test_df["normalized_name"] = test_df.Name.apply(_normalize_name)
    concat_df = pd.concat([train_df, test_df])

    global series_dict
    series_dict = {}
    for i in np.arange(5, 1, step=-1):
        for word, freq in Counter(
            concat_df.normalized_name.drop_duplicates().apply(
                lambda x: " ".join(x.split()[:i])
            )
        ).items():
            if freq >= 3 and freq:
                series_dict[word] = i

    def get_series_name(name):
        for word, _ in reversed(sorted(series_dict.items(), key=lambda x: x[1])):
            if word in name:
                return word
        if name.split()[0] in [
            "the",
            "super",
            "dragon",
            "star",
            "final",
            "mega",
            "shin",
            "world",
            "of",
            "king",
        ]:
            return " ".join(name.split()[:1])
        else:
            return name.split()[0]

    train_df["series_Name"] = train_df.normalized_name.apply(get_series_name)
    test_df["series_Name"] = test_df.normalized_name.apply(get_series_name)


def combination_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    category_cols = [
        "Platform",
        "Genre",
        "norm_Year_of_Release",
        "Rating",
    ]
    for col1, col2 in combinations(category_cols, 2):
        train_df[f"{col1}_{col2}"] = train_df.apply(
            lambda x: f"{str(x[col1])}_{str(x[col2])}", axis=1
        )
        test_df[f"{col1}_{col2}"] = test_df.apply(
            lambda x: f"{str(x[col1])}_{str(x[col2])}", axis=1
        )


def rating_mapping(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Ratingを対象年齢に変換する。
    https://ja.wikipedia.org/wiki/%E3%82%A8%E3%83%B3%E3%82%BF%E3%83%BC%E3%83%86%E3%82%A4%E3%83%B3%E3%83%A1%E3%83%B3%E3%83%88%E3%82%BD%E3%83%95%E3%83%88%E3%82%A6%E3%82%A7%E3%82%A2%E3%83%AC%E3%82%A4%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0%E5%A7%94%E5%93%A1%E4%BC%9A
    """
    rating_map = {
        "E": 6,
        "E10+": 10,
        "M": 17,
        "T": 13,
        "EC": 5,
        "AO": 18,
        "RP": -1,
        "K-A": 6,
    }
    train_df["rating_age"] = train_df.Rating.apply(lambda x: rating_map.get(x, -1))
    test_df["rating_age"] = test_df.Rating.apply(lambda x: rating_map.get(x, -1))


def label_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    le_dict: Dict[str, LabelEncoder] = {},
) -> List[List[pd.DataFrame]]:
    cols = [
        "Genre",
        "Rating",
        "Platform",
    ]
    _train_df = train_df[cols].copy()
    _test_df = test_df[cols].copy()

    for col in cols:
        if _train_df[col].dtype == np.dtype("O"):
            _train_df[col] = _train_df[col].fillna("missing")
            _test_df[col] = _test_df[col].fillna("missing")
        else:
            _train_df[col] = _train_df[col].fillna(-1)
            _test_df[col] = _test_df[col].fillna(-1)

        le = LabelEncoder()
        concatenated = pd.concat([_train_df[col], _test_df[col]], axis=0).reset_index(
            drop=True
        )

        le_dict[col] = le.fit(concatenated)
        _train_df[col] = le.transform(_train_df[col])
        _test_df[col] = le.transform(_test_df[col])

    return [_train_df, _test_df]


def name_processing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[pd.DataFrame]:
    _train_df = pd.DataFrame(index=train_df.index)
    _test_df = pd.DataFrame(index=test_df.index)

    def split_title(normalized_name: str) -> List[str]:
        return re.split(r"[ +|//|&]", normalized_name)

    # 正規化したPublisherごとに良く出てくる単語を抽出
    concat_df = pd.concat([train_df, test_df])
    freq_words: List[str] = []
    counter: Counter = Counter()

    # for words in concat_df.groupby("norm_Publisher").normalized_name.agg(
    #     lambda x: set([__x for _x in x for __x in split_title(_x)])
    # ):
    for words in concat_df.normalized_name:
        counter.update(split_title(words))
    for word, freq in counter.items():
        if freq >= 5 and re.findall("\d+", word) == []:
            freq_words.append(word)

    _train_df["splited_name"] = train_df["normalized_name"].apply(split_title)
    _test_df["splited_name"] = test_df["normalized_name"].apply(split_title)

    _train_df["splited_name"] = _train_df["splited_name"].apply(
        lambda x: list(set(x).intersection(set(freq_words)))
    )
    _test_df["splited_name"] = _test_df["splited_name"].apply(
        lambda x: list(set(x).intersection(set(freq_words)))
    )

    # Multi Binarizer
    _concat_df = pd.concat([_train_df, _test_df])
    mlb = MultiLabelBinarizer()
    mlb.fit(_concat_df["splited_name"])

    column_names = [f"splited_name_{i}" for i in range(len(mlb.classes_))]
    out_train_df = pd.DataFrame(
        mlb.transform(_train_df["splited_name"]), columns=column_names
    )
    out_test_df = pd.DataFrame(
        mlb.transform(_test_df["splited_name"]), columns=column_names
    )

    with open(config.DATA_DIR.joinpath("splited_name.json"), "w") as f:
        json.dump({f"splited_name_{i}": n for i, n in enumerate(mlb.classes_)}, f)

    return [out_train_df, out_test_df]


def add_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[pd.DataFrame]:
    cols = [
        "Critic_Score",
        "Critic_Count",
        "User_Score",
        "User_Count",
        "Year_of_Release",
        "rating_age",
    ]

    _train_df = train_df[cols].copy()
    _test_df = test_df[cols].copy()

    _train_df["User_Score"] = (
        _train_df["User_Score"].replace("tbd", np.nan).astype(np.float32)
    )
    _test_df["User_Score"] = (
        _test_df["User_Score"].replace("tbd", np.nan).astype(np.float32)
    )
    _train_df["User_Score_Sum"] = train_df["User_Score"] * train_df["User_Count"]
    _test_df["User_Score_Sum"] = test_df["User_Score"] * test_df["User_Count"]
    _train_df["Critic_Score_Sum"] = train_df["Critic_Score"] * train_df["User_Count"]
    _test_df["Critic_Score_Sum"] = test_df["Critic_Score"] * test_df["User_Count"]

    return [_train_df, _test_df]


def groupby_target_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[pd.DataFrame]:
    """Global Salesの値をcolumnsごとに集計したもの"""
    _train_df = pd.DataFrame(index=train_df.index)
    _test_df = pd.DataFrame(index=test_df.index)

    cols = [
        "Platform",
        "Genre",
        "Year_of_Release",
        "norm_Year_of_Release",
        "Rating",
        # combination categories
        "Platform_Genre",
        "Platform_norm_Year_of_Release",
        "Platform_Rating",
        "Genre_norm_Year_of_Release",
        "Genre_Rating",
        "norm_Year_of_Release_Rating",
    ]
    for col in cols:
        for t_col in [
            "NA_Sales",
            "JP_Sales",
            "Other_Sales",
            "EU_Sales",
        ]:

            agg_df = train_df.groupby(col)[t_col].agg(["mean", "std", "max", "count"])
            agg_df = agg_df.rename(columns=lambda x: f"agg_{t_col}_by_{col}_{x}")

            agg_df.to_csv(config.DATA_DIR.joinpath(f"agg_df_{t_col}_by_{col}.csv"))
            agg_df = agg_df[agg_df[f"agg_{t_col}_by_{col}_count"] >= 2]

            train_df_with_agg = train_df.merge(agg_df, on=col, how="left")
            test_df_with_agg = test_df.merge(agg_df, on=col, how="left")

            _df = train_df_with_agg.iloc[
                :, train_df_with_agg.columns.str.startswith(f"agg_{t_col}_by_{col}")
            ]
            _train_df = pd.concat([_train_df, _df], axis=1)

            _df = test_df_with_agg.iloc[
                :, test_df_with_agg.columns.str.startswith(f"agg_{t_col}_by_{col}")
            ]
            _test_df = pd.concat([_test_df, _df], axis=1)

    return [_train_df, _test_df]


def groupby_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[pd.DataFrame]:
    out_train_df = pd.DataFrame()
    out_test_df = pd.DataFrame()

    concat_df = pd.concat([train_df, test_df])
    cols = [
        "norm_Developer",
        "norm_Publisher",
        "series_Name",
        "Genre",
        "Platform",
        "Year_of_Release",
        "norm_Year_of_Release",
        "Rating",
        # combination categories
        "Platform_Genre",
        "Platform_norm_Year_of_Release",
        "Platform_Rating",
        "Genre_norm_Year_of_Release",
        "Genre_Rating",
        "norm_Year_of_Release_Rating",
    ]
    for _, col in enumerate(cols):
        _train_df = pd.DataFrame(train_df[col])
        _test_df = pd.DataFrame(test_df[col])

        def p95(x):
            return x.quantile(0.95)

        def p75(x):
            return x.quantile(0.75)

        def p10(x):
            return x.quantile(0.1)

        def p95_minus_p5(x):
            return x.quantile(0.95) - x.quantile(0.05)

        agg_df = concat_df.groupby(col).agg(
            {
                "Name": "count",
                "Genre": pd.Series.nunique,
                "rating_age": ["mean", "median", "count", "sum", p95_minus_p5],
                "Critic_Score": ["mean", "median", "count", "sum", p95_minus_p5],
                "Critic_Count": ["mean", "median", "count", "sum", p95_minus_p5],
                "User_Score": ["mean", "median", "count", "sum", p95_minus_p5],
                "User_Count": ["mean", "median", "count", "sum", p95_minus_p5],
                "Year_of_Release": [
                    p95,
                    p75,
                    "median",
                    p10,
                    p95_minus_p5,
                    "std",
                    "mean",
                ],
            }
        )
        agg_df.columns = [
            f"agg_{col}_{l0}_{l1}"
            for l0, l1 in zip(
                agg_df.columns.get_level_values(0), agg_df.columns.get_level_values(1)
            )
        ]
        agg_df.to_csv(config.DATA_DIR.joinpath(f"agg_{col}.csv"))

        # aggregate先が3以上のもののみを採用する。
        agg_df = agg_df[agg_df[f"agg_{col}_Name_count"] >= 3]

        _train_df = _train_df.merge(agg_df, on=col, how="left").drop([col], axis=1)
        _test_df = _test_df.merge(agg_df, on=col, how="left").drop([col], axis=1)

        out_train_df = pd.concat([out_train_df, _train_df], axis=1)
        out_test_df = pd.concat([out_test_df, _test_df], axis=1)

        assert all(train_df.index == out_train_df.index)
        assert all(test_df.index == out_test_df.index)

    # ============================================
    # === Diff Feature
    # ============================================
    elem_cols = [
        "Year_of_Release",
        "Critic_Score",
        "User_Score",
        "Critic_Count",
        "User_Count",
    ]
    for d_col in cols:
        for e_col in elem_cols:
            out_train_df[f"diff_{d_col}_{e_col}"] = (
                out_train_df[f"agg_{d_col}_{e_col}_mean"] - train_df[e_col]
            )
            out_test_df[f"diff_{d_col}_{e_col}"] = (
                out_test_df[f"agg_{d_col}_{e_col}_mean"] - test_df[e_col]
            )

    return [out_train_df, out_test_df]


def lda_processing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[pd.DataFrame]:
    out_train_df = pd.DataFrame()
    out_test_df = pd.DataFrame()

    concat_df = pd.concat([train_df, test_df])

    for col in ["series_Name", "norm_Publisher", "norm_Developer"]:
        _train_df = pd.DataFrame(train_df[col])
        _test_df = pd.DataFrame(test_df[col])
        aggs = (
            concat_df.groupby(col)
            .agg(
                {
                    "Platform": list,
                    "Year_of_Release": list,
                    "norm_Year_of_Release": list,
                    "Genre": list,
                    "Rating": list,
                    "Platform_Genre": list,
                    "Platform_norm_Year_of_Release": list,
                    "Platform_Rating": list,
                    "Genre_norm_Year_of_Release": list,
                    "Genre_Rating": list,
                    "norm_Year_of_Release_Rating": list,
                }
            )
            .apply(
                lambda x: " ".join(
                    [f"{c}_{v}" for c, values in x.iteritems() for v in values]
                ),
                axis=1,
            )
        )

        # vectorizer = TfidfVectorizer()
        vectorizer = CountVectorizer()
        vectorizer.fit(aggs)
        input_vec = vectorizer.transform(aggs)
        n_components = 10
        lda = LatentDirichletAllocation(
            n_components=n_components,
            learning_method="online",
            random_state=config.SEED,
        )
        svd = TruncatedSVD(n_components=n_components, random_state=config.SEED)
        nmf = NMF(n_components=n_components, random_state=config.SEED)

        lda_vec = lda.fit_transform(input_vec)
        svd_vec = svd.fit_transform(input_vec)
        nmf_vec = nmf.fit_transform(input_vec)
        vec = np.concatenate([lda_vec, svd_vec, nmf_vec], axis=1)
        agg_df = pd.DataFrame(
            vec,
            index=aggs.index,
            columns=[f"lda_{col}_{i}" for i in range(vec.shape[1])],
        )
        agg_df.to_csv(config.DATA_DIR.joinpath(f"lda_{col}.csv"))

        _train_df = _train_df.merge(agg_df, on=col, how="left").drop([col], axis=1)
        _test_df = _test_df.merge(agg_df, on=col, how="left").drop([col], axis=1)

        out_train_df = pd.concat([out_train_df, _train_df], axis=1)
        out_test_df = pd.concat([out_test_df, _test_df], axis=1)

    return [out_train_df, out_test_df]


def rank_by_cols(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[pd.DataFrame]:
    out_train_df = pd.DataFrame(index=train_df.index)
    out_test_df = pd.DataFrame(index=test_df.index)

    cols = [
        "norm_Developer",
        "norm_Publisher",
        "series_Name",
        "Genre",
        "Platform",
        "Rating",
        "Year_of_Release",
        "norm_Year_of_Release",
        # combination categories
        "Platform_Genre",
        "Platform_norm_Year_of_Release",
        "Platform_Rating",
        "Genre_norm_Year_of_Release",
        "Genre_Rating",
        "norm_Year_of_Release_Rating",
    ]
    elem_cols = [
        "Year_of_Release",
        "User_Score",
        "User_Count",
        "Critic_Score",
        "User_Score",
    ]

    for col in cols:
        for e_col in elem_cols:
            out_train_df[f"rank_{col}_{e_col}"] = train_df.groupby(col)[e_col].rank(
                method="first"
            )
            out_train_df[f"rank_pct_{col}_{e_col}"] = train_df.groupby(col)[e_col].rank(
                method="first", pct=True
            )
            out_test_df[f"rank_{col}_{e_col}"] = test_df.groupby(col)[e_col].rank(
                method="first"
            )
            out_test_df[f"rank_pct_{col}_{e_col}"] = test_df.groupby(col)[e_col].rank(
                method="first", pct=True
            )

    return [out_train_df, out_test_df]
