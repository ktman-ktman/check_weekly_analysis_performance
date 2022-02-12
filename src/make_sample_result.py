#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime

import click
import numpy as np
import pandas as pd


def make_sample_result_data(
    from_dateymd: int = 19890101,
    to_dateymd: int = 20211231,
    available_from_dateymd: int = 20100101,
    available_to_dateymd: int = 20210331,
    freq: str = "W-MON",
) -> pd.DataFrame:
    from_date = datetime.datetime.strptime(str(from_dateymd), "%Y%m%d").date()
    to_date = datetime.datetime.strptime(str(to_dateymd), "%Y%m%d").date()
    available_from_date = datetime.datetime.strptime(
        str(available_from_dateymd), "%Y%m%d"
    ).date()
    available_to_date = datetime.datetime.strptime(
        str(available_to_dateymd), "%Y%m%d"
    ).date()

    # ベースのdfを作成
    df = pd.DataFrame(pd.date_range(from_date, to_date, freq=freq), columns=["date"])

    # 乱数のデータを作成
    df = df.assign(
        drtnp=np.random.normal(
            loc=0.05 / 52, scale=0.2 / np.sqrt(52), size=df.shape[0]
        ),  # 平均: 0.05，標準偏差 0.2の正規分布
        drtnf=np.random.normal(
            loc=0.05 / 52, scale=0.2 / np.sqrt(52), size=df.shape[0]
        ),  # 平均: 0.05，標準偏差 0.2の正規分布
        prob=np.random.rand(df.shape[0]),  # 0 ~ 1の一様分布
        training_loss=np.random.rand(df.shape[0]),  # 0 ~ 1の一様分布
        training_acc=np.random.rand(df.shape[0]),  # 0 ~ 1の一様分布
        validation_loss=np.random.rand(df.shape[0]),  # 0 ~ 1の一様分布
        validation_acc=np.random.rand(df.shape[0]),  # 0 ~ 1の一様分布
    )
    # リターンを合わせるため固定
    np.random.seed(1)
    df.loc[:, "drtnp"] = np.random.normal(
        loc=0.05 / 52,
        scale=0.2 / np.sqrt(52),
        size=df.shape[0],
    )
    df.loc[:, "drtnf"] = df["drtnp"].shift(-1)

    # データがない期間はNAで埋める
    mask1 = df["date"] >= "2010/1/14"
    mask2 = df["date"] <= "2021/3/29"
    mask = mask1 & mask2
    coln_l = [x for x in df.columns.tolist() if x not in ["date", "drtnp", "drtnf"]]
    df.loc[~mask, coln_l] = pd.NA
    return df


@click.command()
@click.argument("output_filename", type=str)
def main(output_filename: str) -> None:
    ofn = f"../data/result/{output_filename}.csv"

    df = make_sample_result_data()
    df.to_csv(ofn, index=False)


if __name__ == "__main__":
    main()
