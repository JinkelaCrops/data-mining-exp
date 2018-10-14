# 生成数据6prize, 100天, 1000000users, 支用概率是用户活跃度+奖品等级*随机数+随机数 > 1.5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# import lightgbm as lgb

# data size
user_num = 1000000
times = 1000

data = pd.DataFrame(index=np.arange(user_num) + user_num * 10)
data["user_level"] = np.random.rand(user_num)
data["user_level__1"] = data["user_level"] ** (-1)
data["user_level__2"] = data["user_level"] ** (-2)
data["user_level_2"] = data["user_level"] ** 2
data["user_level_3"] = data["user_level"] ** 3
data["channel_level"] = np.random.rand(user_num)
data["channel_level__1"] = data["channel_level"] ** (-1)
data["channel_level__2"] = data["channel_level"] ** (-2)
data["channel_level_2"] = data["channel_level"] ** 2
data["channel_level_3"] = data["channel_level"] ** 3
data["message_level"] = np.random.rand(user_num)
data["message_level__1"] = data["message_level"] ** (-1)
data["message_level__2"] = data["message_level"] ** (-2)
data["message_level_2"] = data["message_level"] ** 2
data["message_level_3"] = data["message_level"] ** 3
data["prize_id"] = np.random.randint(36, 42, user_num)
data["date"] = np.random.randint(0, times, user_num)
data["prize_level"] = data["prize_id"].map(dict(zip(np.arange(36, 42), np.arange(0, 6))))
data["user_prize"] = data["user_level"] * data["prize_level"]
data["channel_prize"] = data["channel_level"] * data["prize_level"]
data["message_prize"] = data["message_level"] * data["prize_level"]
# function
data["use_prize"] = 1 \
                    + 0.001 * data["channel_level"] \
                    + 0.001 * data["user_level"] \
                    + 0.001 * data["message_level"] \
                    + 0.2 * data["channel_level"] * data["prize_level"] \
                    + 0.2 * data["user_level"] * data["prize_level"] \
                    + 0.2 * data["message_level"] * data["prize_level"] \
                    + (data["user_level"] + 3 * np.random.rand(user_num)) / (data["prize_level"] ** 2 + 10) \
                    + (data["channel_level"] + 3 * np.random.rand(user_num)) / (data["prize_level"] ** 2 + 10) \
                    + (data["message_level"] + 3 * np.random.rand(user_num)) / (data["prize_level"] ** 2 + 10) \
                    + 2 * np.random.rand(user_num) / (data["user_level"] + 1) \
                    + 2 * np.random.rand(user_num) / (data["channel_level"] + 1) \
                    + 2 * np.random.rand(user_num) / (data["message_level"] + 1) \
                    + np.random.rand(user_num)
plt.hist(data["use_prize"], bins=1000)
plt.show()
threshold = (0.5 * data["use_prize"].max() - data["use_prize"].min()) + data["use_prize"].min()
data["y"] = (data["use_prize"] > threshold) * 1

data_inner_rate = data.groupby(["prize_id"])["y"].mean()
print("奖品核销率分布", data_inner_rate)
print("内部标准差", np.std(data_inner_rate))

# std
date_rate = data.groupby(["date"])["y"].mean()
plt.hist(date_rate, bins=40)
plt.show()
print("标准差", np.std(date_rate))

# OOT
data["tag"] = (data["date"] >= int(times * (1 - 0.001))).map({True: "test", False: "train"})
data = data.sort_values(by=["date"])

# y ~ user_level + prize_level

# data.to_csv("data/my_data.csv", encoding="utf8", index=None)
with open("data/my_data.x", "wb") as f:
    pickle.dump(data, f)

data["const"] = 1
y = data["use_prize"]
x = data[["const", "prize_level", "user_prize", "channel_prize", "message_prize",
          "user_level", "user_level__1", "user_level__2", "user_level_2", "user_level_3",
          "channel_level", "channel_level__1", "channel_level__2", "channel_level_2", "channel_level_3",
          "message_level", "message_level__1", "message_level__2", "message_level_2", "message_level_3"]]
params = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
r2 = 1 - np.sum((y - x.dot(params)) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(r2)
