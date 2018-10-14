import pickle
import numpy as np
from xgboost import DMatrix, train

with open("data/my_data.x", "rb") as f:
    data = pickle.load(f)
print(list(data.columns))
# y: ["y"], x: ["user_level", "prize_level"]
train_data = data[data["tag"] == "train"]
test_data = data[data["tag"] == "test"]
x_label = ["prize_level", "user_prize", "channel_prize", "message_prize",
           "user_level", "user_level__1", "user_level__2", "user_level_2", "user_level_3",
           "channel_level", "channel_level__1", "channel_level__2", "channel_level_2", "channel_level_3",
           "message_level", "message_level__1", "message_level__2", "message_level_2", "message_level_3"
           ]
y_label = ["y"]

xgb_params1 = {
    'booster': 'gbtree',
    'eta': 0.3,
    'gamma': 1.0,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'max_depth': 6,
    'silent': 1,
    'eval_metric': 'auc',
    'save_period': 0,
    "learning_rates": 0.01
}

xgbTrain = DMatrix(train_data[x_label], label=train_data[y_label])
# generate eval data
xgbTrain_eval = DMatrix(test_data[x_label], label=test_data[y_label])
evallist = [(xgbTrain, 'train'), (xgbTrain_eval, 'eval')]
# train model
# xgb_rank_params1加上 evals 这个参数会报错，还没找到原因
# rankModel = train(xgb_rank_params1,xgbTrain,num_boost_round=10)
rankModel = train(xgb_params1, xgbTrain, num_boost_round=30, evals=evallist)
# test dataset
# test
y_pred = rankModel.predict(xgbTrain_eval)

print(rankModel.get_fscore())
# acc
acc = np.mean(test_data[y_label[0]] == (y_pred > 0.5) * 1)
print(acc)
