import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_folds(df: pd.DataFrame, random_state: int = 21, save = None) -> pd.DataFrame:
    df['fold'] = 0

    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(df, df.object_id, df.group)):
        ids = df.loc[test_index].img_name.tolist()
        df['fold'] = df.apply(lambda x: i if x.img_name in ids else x.fold, axis=1)

    if save is not None:
        df.to_csv(save, index=False)

    return df
