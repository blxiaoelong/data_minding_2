import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori  # 生成频繁项集
from mlxtend.frequent_patterns import association_rules  # 生成强关联规则

path = ['anonymous-msweb.data','anonymous-msweb.test']
attr_dict = {}
user_case = []
def Getdata(path):
    c = []
    with open(path, 'r+') as path:
        for line in path:
            token = line.split(',')
            if token[0] == 'A':
                attr_dict[token[1]] = ''.join((token[3][1:-1],token[4][1:-2]))
            elif token[0] == 'C':
                user_case.append(c)
                c = []
            elif token[0] == 'V':
                c.append(attr_dict[token[1]])
    return user_case

if __name__ == '__main__':
    dataSet = Getdata(path[0])
    # print(dataSet)
    column_list = []
    for var in dataSet:
        column_list = set(column_list) | set(var)
    print('转换原数据到0-1矩阵')
    data = pd.DataFrame(np.zeros((len(dataSet), 285)), columns=column_list)
    for i in range(len(dataSet)):
        for j in dataSet[i]:
            data.loc[i, j] += 1
    # apriori算法
    frequent_itemsets = apriori(data, min_support=0.02, use_colnames=True)
    print(pd.DataFrame(frequent_itemsets))
    pd.DataFrame(frequent_itemsets).to_csv('frequent_itemsets.csv')
    # 生成关联准则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
    print(pd.DataFrame(rules))
    pd.DataFrame(rules).to_csv('rules.csv')
