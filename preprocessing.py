import os
import pandas as pd
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from utils import complete_timestamp, standardize_kpi


def process_kpi_data(train_path, test_path, out_path, standard=False, filled_type="linear"):
    """
    Preprocess KPI dataset.
    Fill the missing data, then save the training and testing data to the same .csv file.

    Args:
        train_path: path of training data.
        test_path: path of testing data.
        out_path: path to store the processed data.
        standard: whether to standardize the curve. [True/False]
        filled_type: method type of filling the missing data. [linear/periodic]
    """
    train_df = read_data(train_path)[['timestamp', 'value', 'label', 'KPI ID']]
    test_df = read_data(test_path)[['timestamp', 'value', 'label', 'KPI ID']]

    data_df = pd.DataFrame(columns=['timestamp', 'value', 'label', 'KPI ID', "missing", "is_test"])
    group_list = [train_df.groupby("KPI ID"), test_df.groupby("KPI ID")]

    mean_dict = {}
    std_dict = {}
    for i in range(len(group_list)):
        for name, group in group_list[i]:
            print(name)
            temp_df = pd.DataFrame(columns=['timestamp', 'value', 'label', 'KPI ID', "missing", "is_test"])
            timestamp = group["timestamp"].values
            value = group["value"].values
            label = group["label"].values

            timestamp, missing, (value, label), interval, max_miss_num = complete_timestamp(timestamp, (value, label))

            # Standardize the training and testing data.
            if standard:
                if i == 0:  # for training data, calculate the mean and std values
                    value, mean, std = standardize_kpi(value, excludes=np.logical_or(label, missing))
                    mean_dict[name] = mean
                    std_dict[name] = std
                else:  # for testing data, use the mean and std values of training data
                    mean = mean_dict[name]
                    std = std_dict[name]
                    value, _, _ = standardize_kpi(value, mean=mean, std=std)

            label[np.isnan(label)] = 0  # replace nan of label array with zero
            print("max_miss_num: ", max_miss_num)

            temp_df['timestamp'], temp_df["missing"], temp_df["value"], temp_df["label"] = \
                timestamp, missing, value, label
            temp_df["KPI ID"], temp_df["is_test"] = name, i  # i: identifier for training or test data

            # for filling
            period = 1440 * 60 // interval
            length = len(value)
            num_padding = (length // period + 1) * period - length
            if filled_type == "linear":
                temp_df['value'].interpolate(method='linear', inplace=True)  # linear interpolation
            elif filled_type == "periodic":  # replace long missing seg with the last periodic values
                tmp_value = np.concatenate((value, np.full([num_padding], np.nan)))
                tmp_2d_array = np.reshape(tmp_value, (-1, period))
                nan_num = np.sum(tmp_2d_array != tmp_2d_array, axis=1)  # calculate the np.nan num for each line
                for k in range(tmp_2d_array.shape[0]):
                    if nan_num[k] > 5:
                        filled_idx = np.where(np.isnan(tmp_2d_array[k]))[0]
                        filled_idx_list = [list(g) for g in mit.consecutive_groups(filled_idx)]
                        for m in range(len(filled_idx_list)):
                            idx_seg = np.array(filled_idx_list[m])
                            if 5 < len(idx_seg) < 0.6 * period:  # balance the mean value
                                mean_diff = np.nanmean(tmp_2d_array[k]) - np.nanmean(tmp_2d_array[k - 1])
                                tmp_2d_array[k, idx_seg] = tmp_2d_array[k - 1, idx_seg] + mean_diff / 2
                            elif len(idx_seg) > 0.6 * period:
                                tmp_2d_array[k, idx_seg] = tmp_2d_array[k - 1, idx_seg]
                flatten_value = tmp_2d_array.flatten()[: length]
                temp_df['value'] = flatten_value
                # Finally, do linear interpolation for short segments
                temp_df['value'].interpolate(method='linear', inplace=True)
            else:
                raise TypeError("This filled type is not available！")
            data_df = pd.concat([data_df, temp_df], ignore_index=True)
            # show_filled_data(name, filled_type, temp_df['value'].values, temp_df['missing'].values)

    data_df.to_csv(out_path, index=False)
    # with open("mean_std_info.txt", "w") as f:
    #     f.write(str({"mean": mean_dict, "std": std_dict}))


def show_filled_data(kip_id, fill_type, data, missing, label=None):
    missing_group = [list(g) for g in mit.consecutive_groups(np.where(missing)[0])]
    missing_segs = [(g[0], g[-1]) if g[0] != g[-1] else (g[0] - 1, g[0] + 1) for g in missing_group]

    _len = len(data)
    xs = np.linspace(0, _len - 1, _len)
    plt.figure(figsize=(9, 6))
    plt.title("id: {}, type: {}".format(kip_id, fill_type))
    plt.xticks([])
    plt.yticks([])
    plt.plot(xs, data, "mediumblue")
    for seg in missing_segs:
        seg_x = np.linspace(seg[0], seg[1], seg[1] - seg[0] + 1).astype(dtype=int)
        plt.plot(seg_x, data[seg_x], color="g")

    if label is not None:
        label_group = [list(g) for g in mit.consecutive_groups(np.where(label)[0])]
        label_segs = [(g[0], g[-1]) if g[0] != g[-1] else (g[0] - 1, g[0] + 1) for g in label_group]
        for seg in label_segs:
            seg_x = np.linspace(seg[0], seg[1], seg[1] - seg[0] + 1).astype(dtype=int)
            plt.plot(seg_x, data[seg_x], color="r")
    plt.show()


def read_data(path):
    if path.endswith(".hdf"):
        df = pd.read_hdf(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise TypeError("Current file type is Not available!")

    return df


if __name__ == "__main__":
    train_path = "./data/AIOps/phase2_train.csv"
    test_path = "./data/AIOps/phase2_ground_truth.hdf"
    out_path = "./data/AIOps/total_data.csv"
    process_kpi_data(train_path, test_path, out_path, standard=False, filled_type="periodic")
