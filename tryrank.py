import pandas as pd
import numpy as np
from xgboost import DMatrix, train
from XGBRanker import XGBRanker

xgb_rank_params1 = {
    # 'booster': 'gbtree',
    'eta': 1,
    # 'gamma': 1.0,
    # 'min_child_weight': 0.1,
    'objective': 'rank:pairwise',
    # 'eval_metric': 'merror',
    'max_depth': 6,
    # 'num_boost_round': 10,
    'silent': 1,
    'eval_metric': 'ndcg',
    'save_period': 0
}
xgb_rank_params2 = {'max_depth': 2,
                    'eta': 1,
                    'silent': 1,
                    'objective': 'rank:pairwise',
                    # 'nthread': 4,
                    'eval_metric': 'ndcg'}
# generate training dataset
# 一共2组*每组3条，6条样本，特征维数是2
n_group = 2
n_choice = 3
dtrain = np.random.uniform(0, 100, [n_group * n_choice, 2])
# numpy.random.choice(a, size=None, replace=True, p=None)
dtarget = np.array([np.random.choice([0, 1, 2], 3, False) for i in range(n_group)]).flatten()
# n_group用于表示从前到后每组各自有多少样本，前提是样本中各组是连续的，[3，3]表示一共6条样本中前3条是第一组，后3条是第二组
dgroup = np.array([n_choice for i in range(n_group)]).flatten()
# concate Train data, very import here !
xgbTrain = DMatrix(dtrain, label=dtarget)
xgbTrain.set_group(dgroup)
# generate eval data
dtrain_eval = np.random.uniform(0, 100, [n_group * n_choice, 2])
xgbTrain_eval = DMatrix(dtrain_eval, label=dtarget)
xgbTrain_eval.set_group(dgroup)
evallist = [(xgbTrain, 'train'), (xgbTrain_eval, 'eval')]
# train model
# xgb_rank_params1加上 evals 这个参数会报错，还没找到原因
# rankModel = train(xgb_rank_params1,xgbTrain,num_boost_round=10)
rankModel = train(xgb_rank_params2, xgbTrain, num_boost_round=20, evals=evallist)
# test dataset
dtest = np.random.uniform(0, 100, [n_group * n_choice, 2])
dtestgroup = np.array([n_choice for i in range(n_group)]).flatten()
xgbTest = DMatrix(dtest)
xgbTest.set_group(dgroup)
# test
print(rankModel.predict(xgbTest))

# -----------------------------------------
CASE_NUM = 20
GROUPS_NUM = 4
if CASE_NUM % GROUPS_NUM != 0:
    raise ValueError('Cases should be splittable into equal groups.')
X_features = np.random.rand(CASE_NUM, 4)
y = np.random.randint(5, size=CASE_NUM)
X_groups = np.arange(0, GROUPS_NUM).repeat(CASE_NUM / GROUPS_NUM)
print("X=" + str(X_features))
print("y=" + str(y))

X = np.concatenate([X_groups[:, None], X_features], axis=1)
ranker = XGBRanker(n_estimators=150, learning_rate=0.1, subsample=0.9)
ranker.fit(X, y, eval_metric=['ndcg', 'map@5-'])
y_predict = ranker.predict(X)

print("predict:" + str(y_predict))
print("type(y_predict):" + str(type(y_predict)))

# ------------------------------------------
