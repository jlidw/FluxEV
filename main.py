import os
import os.path as osp
import time
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score, recall_score, precision_score
import argparse
import numba

import sys
sys.path.append('..')  # import the upper directory of the current file into the search path
from spot_pipe import SPOT
from eval_methods import adjust_predicts


def calc_ewma(input_arr, alpha=0.2, adjust=True):
    """
    Here, we use EWMA as a simple predictor.

    Args:
        input_arr: 1-D input array
        alpha: smoothing factor, (0, 1]
        adjust:
            (1) When adjust=True(faster), the EW function is calculated using weights w_i=(1-alpha)^i;
            (2) When adjust=False, the exponentially weighted function is calculated recursively.
    Returns:
        Exponentially Weighted Average Value
    """
    arr_len = len(input_arr)
    if adjust:
        power_arr = np.array(range(len(input_arr)-1, -1, -1))
        a = np.full(arr_len, 1-alpha)
        weight_arr = np.power(a, power_arr)
        ret = np.sum(input_arr * weight_arr) / np.sum(weight_arr)
    else:
        ret_arr = [input_arr[0]]
        for i in range(1, arr_len):
            temp = alpha * input_arr[i] + (1 - alpha) * ret_arr[-1]
            ret_arr.append(temp)
        ret = ret_arr[-1]
    return ret


@numba.jit(nopython=True)
def calc_ewma_v2(input_arr, alpha=0.2):
    arr_len = len(input_arr)
    ret_arr = [input_arr[0]]
    for i in range(1, arr_len):
        temp = alpha * input_arr[i] + (1 - alpha) * ret_arr[-1]
        ret_arr.append(temp)
    ret = ret_arr[-1]
    return ret


def calc_first_smooth(input_arr):
    return max(np.nanstd(input_arr) - np.nanstd(input_arr[:-1]), 0)  # if std_diff < 0, return 0


def calc_second_smooth(input_arr):
    return max(np.nanmax(input_arr) - np.nanmax(input_arr[:-1]), 0)  # if max_diff < 0, return 0


def detect(data_arr, train_len, period, smoothing=2,
           s_w=10, p_w=7, half_d_w=2, q=0.001,
           estimator="MOM"):
    """
    Args:
        data_arr: 1-D data array.
        train_len: data length for training.
        period: data period, usually indicate the point num of one day;
                one-min level: 1440, one-hour level: 24.
        smoothing: number of smoothing operations;
                   1->only first-step smoothing, 2->two-step smoothing.
        s_w: sequential window size for detecting anomaly.
        p_w: periodic window size for detecting anomaly.
        half_d_w: half window size of handling data drift, for detecting anomaly.
        q: risk coefficient of SPOT for detecting anomaly;
           usually between 10^-3 and 10^-5 to have a good performance.
        estimator: estimation method for data distribution in SPOT, "MOM" or "MLE".
    Returns:
        alarms: detection results, 0->normal, 1->abnormal.
    """
    data_len = len(data_arr)
    spot = SPOT(q, estimator=estimator)  # create detector

    d_w = half_d_w * 2

    # Calculate the start index to extract anomaly features
    fs_idx = s_w * 2  # start index for first smoothing
    fs_lm_idx = fs_idx + d_w  # start index for local max array of first smoothing
    ss_idx = fs_idx + half_d_w + period * (p_w - 1)  # start index for second smoothing

    pred_err = np.full(data_len, np.nan)  # prediction error array (predictor: ewma)
    fs_err = np.full(data_len, np.nan)  # first smoothing error array
    fs_err_lm = np.full(data_len, np.nan)  # local max array for the first smoothing error
    ss_err = np.full(data_len, np.nan)  # second smoothing error array

    th, alarms = [], []
    if smoothing == 1:
        for i in range(s_w, data_len):
            # calculate the predicted value Pi and prediction error Ei
            Pi = calc_ewma_v2(data_arr[i - s_w: i])
            Ei = data_arr[i] - Pi
            pred_err[i] = Ei

            # first smoothing
            if i >= fs_idx:
                FSEi = calc_first_smooth(pred_err[i - s_w: i + 1])  # fixed index
                fs_err[i] = FSEi

            # SPOT Detection
            if i == train_len - 1:  # initialize SPOT detector using training data
                init_data = fs_err[fs_idx: i + 1]
                spot.fit(init_data)
                spot.initialize()

            if i >= train_len:  # detect the testing point one by one
                # th_s: the current threshold(dynamic); alarm_s: 0->normal, 1->abnormal
                th_s, alarm_s = spot.run_step(fs_err[i])  # apply the detection

                th.append(th_s)
                alarms.append(alarm_s)

    elif smoothing == 2:
        for i in range(s_w, data_len):
            # calculate the predicted value Pi and prediction error Ei
            Pi = calc_ewma_v2(data_arr[i - s_w: i])
            Ei = data_arr[i] - Pi
            pred_err[i] = Ei

            if i >= fs_idx:
                # the first smoothing
                FSEi = calc_first_smooth(pred_err[i - s_w: i + 1])  # fixed index
                fs_err[i] = FSEi

                # extract the local max value
                if i >= fs_lm_idx:
                    FSEi_lm = max(fs_err[i - d_w: i + 1])
                    fs_err_lm[i - half_d_w] = FSEi_lm  # fixed index

                # the second smoothing
                if i >= ss_idx:
                    tem_arr = np.append(fs_err_lm[i - period * (p_w - 1): i: period], fs_err[i])
                    SSEi = calc_second_smooth(tem_arr)
                    ss_err[i] = SSEi

            # SPOT Detection
            if i == train_len - 1:  # initialize SPOT detector using training data
                init_data = ss_err[ss_idx: i + 1]
                spot.fit(init_data)
                spot.initialize()

            if i >= train_len:  # detect the testing point one by one
                # th_s: the current threshold(dynamic); alarm_s: 0->normal, 1->abnormal
                th_s, alarm_s = spot.run_step(ss_err[i])  # apply the detection

                # if detect an anomaly, update its features;
                # avoid affecting feature extraction of subsequent points
                if alarm_s:
                    fs_err[i] = np.nan
                    FSEi_lm = max(fs_err[i - d_w: i + 1])
                    fs_err_lm[i - half_d_w] = FSEi_lm

                th.append(th_s)
                alarms.append(alarm_s)

    alarms = np.array(alarms)
    return alarms


def read_yahoo_data(path):
    file_name = path.split("/")[-1][:-4]
    dir_id = int(path.split("/")[-2][1])

    if dir_id < 3:
        timestamp_col = "timestamp"
        value_col = "value"
        label_col = "is_anomaly"
    else:
        timestamp_col = "timestamps"
        value_col = "value"
        label_col = "anomaly"

    df = pd.read_csv(path)[[timestamp_col, value_col, label_col]]
    # convert to int dtype
    df[[timestamp_col, label_col]] = df[[timestamp_col, label_col]].astype(int)
    df = df.rename(columns={timestamp_col: "timestamp",
                            value_col: "value",
                            label_col: "label"})

    return df, file_name, dir_id


def main_yahoo(args, data_dir):
    ret_file_path = osp.join(data_dir, args.ret_file).format(args.estimator,
                                                             args.s_w, args.p_w,
                                                             args.half_d_w, args.q)

    file_list = []
    for _id in [1, 2, 3, 4]:
        sub_dir = osp.join(data_dir, "A{}Benchmark".format(_id))
        file_list += glob(sub_dir + "/*.csv")

    y_true, y_pred = [], []
    for _path in file_list:
        data_df, file_name, dir_id = read_yahoo_data(_path)

        print(file_name)
        # timestamp = data_df["timestamp"].values  # timestep array
        value = data_df["value"].values  # data array
        label = data_df["label"].values  # label array

        period = 24  # hour-level data

        if not args.train_len:
            train_len = len(value) // 2
        else:
            train_len = args.train_len

        if dir_id == 2:
            smoothing = 1
        else:
            smoothing = 2

        label_test = label[train_len:]
        alarms = detect(value, train_len, period, smoothing,
                        args.s_w, args.p_w, args.half_d_w, args.q,
                        estimator=args.estimator)

        ret_test = adjust_predicts(predict=alarms, label=label_test, delay=args.delay)

        y_true.append(label_test)
        y_pred.append(ret_test)

    y_true_arr, y_pred_arr = np.concatenate(y_true), np.concatenate(y_pred)
    f_score = f1_score(y_true_arr, y_pred_arr)
    recall = recall_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr)

    with open(ret_file_path, "a") as f:
        f.write("Total F1/Recall/Precision score: {}, {}, {}\n".format(f_score, recall, precision))


def main_kpi(args, base_dir, data_path):
    ret_dir = osp.join(base_dir, "results")
    ret_file_path = osp.join(ret_dir, args.ret_file).format(args.estimator,
                                                             args.s_w, args.p_w,
                                                             args.half_d_w, args.q)
    if not osp.exists(ret_dir):
        os.makedirs(ret_dir)

    # read data and convert several data type to int
    data_df = pd.read_csv(data_path)
    data_df[["timestamp", "label", "missing", "is_test"]] = \
        data_df[["timestamp", "label", "missing", "is_test"]].astype(int)

    y_true, y_pred = [], []
    for name, group in data_df.sort_values(by=["KPI ID", "timestamp"], ascending=True).groupby("KPI ID"):
        print(name)

        group.reset_index(drop=True, inplace=True)
        timestamp = group["timestamp"].values
        value = group["value"].values
        label = group["label"].values
        missing = group["missing"].values

        if not args.train_len:
            train_len = sum(group["is_test"].values == 0)
        else:
            train_len = args.train_len

        interval = timestamp[1] - timestamp[0]
        period = 1440 * 60 // interval

        smoothing = 2
        label_test = label[train_len:]
        test_missing = missing[train_len:]
        alarms = detect(value, train_len, period, smoothing,
                        args.s_w, args.p_w, args.half_d_w, args.q,
                        estimator=args.estimator)

        alarms[np.where(test_missing == 1)] = 0  # set the results of missing points to 0
        ret_test = adjust_predicts(predict=alarms, label=label_test, delay=args.delay)

        y_true.append(label_test)
        y_pred.append(ret_test)

    y_true_arr, y_pred_arr = np.concatenate(y_true), np.concatenate(y_pred)

    f_score = f1_score(y_true_arr, y_pred_arr)
    recall = recall_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr)

    with open(ret_file_path, "a") as f:
        f.write("Total F1/Recall/Precision score: {}, {}, {}\n".format(f_score, recall, precision))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Detection of FluxEV")
    parser.add_argument('--dataset', type=str, default='Yahoo')
    parser.add_argument('--delay', type=int, default=7,
                        help="delay point num for evaluation")
    parser.add_argument('--q', type=float, default=0.003,
                        help="risk coefficient for SPOT")

    parser.add_argument('--s_w', type=int, default=10,
                        help="sequential window size "
                             "to extract the local fluctuation and do the first-step smoothing")
    parser.add_argument('--p_w', type=int, default=5,
                        help="periodic window size to do the second-step smoothing")
    parser.add_argument('--half_d_w', type=int, default=2,
                        help="half window size for handling data drift")

    parser.add_argument('--estimator', type=str, default="MOM",
                        help="estimation method for SPOT, 'MOM' or 'MLE'")
    parser.add_argument('--train_len', type=int, default=None,
                        help="data length for training (initialize SPOT), "
                             "if None(default), the program will set it as the half of the data length")

    parser.add_argument('--ret_file', type=str, default='{}-s{}-p{}-d{}-q{}-new12.txt')

    Flags = parser.parse_args()
    if Flags.dataset == "KPI":
        base_dir = "./data/AIOps/"
        data_path = osp.join(base_dir, "total_data.csv")
        Flags.delay = 7
        Flags.q = 0.003
        main_kpi(Flags, base_dir, data_path)
    elif Flags.dataset == "Yahoo":
        data_dir = "./data/Yahoo"
        Flags.delay = 3
        Flags.q = 0.001
        main_yahoo(Flags, data_dir)

