"""-------------------- 配置环境 --------------------"""
# 导入需要的包
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

from surprise import SVD
from surprise import Dataset
from surprise import Reader
import pandas as pd
import os
from scipy.stats import spearmanr

import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark import SQLContext,HiveContext
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Bucketizer
from pyspark.ml import Pipeline, Transformer, Model
from pyspark.ml.pipeline import PipelineModel

from pyhive import hive
import pandas as pd


#连接云超的数据库,端口号，hive:10001,impala:21051
def get_data_from_hive(query):
    conn=hive.connect(host='10.1.53.19',port=21051,username='songyuanchen',password='3w9USIgKHONmhGw',auth='LDAP')
    cur=conn.cursor()
    cur.execute(query)
    data=cur.fetchall()
    columnDes=cur.description #获取连接对象的描述信息
    columnNames=[columnDes[i][0] for i in range(len(columnDes))]
    df=pd.DataFrame([list(i) for i in data],columns=columnNames)
    cur.close()
    return df

os.system("source setpython_spark spark2 python3.5")
os.environ["PYSPARK_PYTHON"]='/usr/bin/python3.5'
os.environ["PYSPARK_DRIVER_PYTHON"]='/usr/bin/python3.5'

spark=SparkSession.builder.appName("new_peizhi") \
    .master("yarn") \
    .config('spark.executor.instances',5) \
    .config('spark.executor.cores', 20) \
    .config("spark.executor.memory", '5G') \
    .config("spark.port.maxRetries", 100) \
    .config("spark.driver.maxResultSize", '4G') \
    .config("spark.serializer", 'org.apache.spark.serializer.KryoSerializer') \
    .config('spark.driver.memory','4G') \
    .config('spark.default.parallelism',60) \
    .config("spark.shuffle.file.buffer", '128k') \
    .config("spark.reducer.maxSizeInFlight", '96m') \
    .config("spark.dynamicAllocation.enabled", False)\
    .enableHiveSupport() \
    .getOrCreate()

"""-------------------- 获取每家门店对在售商品的评分 --------------------"""
# 获取每家门店对每种商品的评分，评分已经计算存入hive中
# 评分的计算方法是把每家门店的商品按销量升序排列，以累积分位数*5作为评分
query=r"SELECT shop_id,goods_id,sum(ratio) over(PARTITION BY shop_id ORDER BY row_num ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM shushi_tmp.wanglinhan_matrix_factorization_shop_goods_ratio_order_big"
shop_goods_ratio_cumsum=get_data_from_hive(query)
shop_goods_rating=shop_goods_ratio_cumsum
shop_goods_rating.columns = ['shop_id','goods_id','rating'] # 对数据进行更名便于后续调用
shop_goods_rating['rating']=shop_goods_rating['rating']*5 # 整列*5，将评分映射到0-5之间



"""-------------------- 对每家门店进行推荐 --------------------"""
def get_top_n(predictions, n=10):
    """获得对每家门店的Top-N商品推荐

    Args:
        predictions(list of Prediction objects): 调用推荐算法之后得到的结果
        n(int): 为每个门店推荐的商品个数，默认是10个

    Returns:
    一个字典，键是shop_id，值是由列表构成的元组
    """

    # 首先把预测值和用户相对应
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # 然后将预测值排序，选出k个评分最高的
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# 需要一个reader来处理数据，需要指定评分范围
reader = Reader(rating_scale=(0,5))

# 用reader导入并处理之前计算得到的评分数据
data = Dataset.load_from_df(shop_goods_rating[['shop_id','goods_id','rating']], reader)

# 生成训练集，并拟合SVD模型
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# 预测所有门店对所有商品的评分
testset = trainset.build_testset() # 在原始数据集中出现过的商品评分
testset1 = trainset.build_anti_testset() # 每家门店没有售卖过的商品
testset.extend(testset1) # 二者组合可得到所有门店和所有商品的组合
predictions = algo.test(testset) # 用分解矩阵来拟合得到预测结果

# 得到为每家门店推荐的评分排名前2000的商品及其评分
top_n = get_top_n(predictions, n=2000)



"""-------------------- 通过比较两种相关性对推荐算法的结果进行有效性检验 --------------------"""
# 计算相关性检验指标的大循环
query=r"SELECT DISTINCT(shop_id) from shushi_tmp.shop_goods_5_6_last_7_month_avg_amt_abe_effect_9M5R_to_9010_angelo_20210619"
shop_id_df=get_data_from_hive(query)

# 得到所有商品的列表
shop_id_list=[]
for i in range(len(shop_id_df)):
    shop_id_list.append(shop_id_df.iloc[i,0])

# corr_record用来记录每个门店的两种相关性
corr_record=pd.DataFrame(columns=('shop_id','corr1','corr2'))
count=0
for j in range(len(shop_id_list)):

    # 得到当前计算的门店的名称shop_name
    shop_name_pre=shop_id_list[j]
    shop_name='{!r}'.format(shop_name_pre)
    recommendation_tuple_list=top_n[shop_name_pre]

    # 取出当前计算门店在售的商品
    query_pre=r"select goods_id from shushi_tmp.shop_goods_5_6_last_7_month_avg_amt_abe_effect_9M5R_to_9010_angelo_20210619 where shop_id = "+ shop_name
    query=query_pre
    goods_one_shop=get_data_from_hive(query)

    # 以列表的形式获得商品
    goods_list=[]
    for i in range(len(goods_one_shop)):
        goods_list.append(goods_one_shop.iloc[i,0])

    # 获得推荐列表和在售商品的交集
    intersection_goods_list=[]
    intersection_goods_score=[]
    for i in recommendation_tuple_list:
        if i[0] in goods_list:
            intersection_goods_list.append(i[0])
            intersection_goods_score.append(i[1])
    df_dict={"goods_id":intersection_goods_list,
            "score":intersection_goods_score}
    goods_score_df=pd.DataFrame(df_dict)


    # 取当前计算门店的销售数据
    query_pre=r'SELECT goods_id,sales_amt from shushi_tmp.shop_goods_5_6_last_7_month_avg_amt_abe_effect_9M5R_to_9010_angelo_20210619 WHERE shop_id = '+shop_name
    query=query_pre
    goods_sales_amt_one_shop=get_data_from_hive(query)

    # 取所有门店的销售数据
    query=r"SELECT goods_id,sum(sales_amt) from shushi_tmp.shop_goods_5_6_last_7_month_avg_amt_abe_effect_9M5R_to_9010_angelo_20210619 GROUP BY goods_id"
    goods_sales_amt_all=get_data_from_hive(query)


    # 第一个相关性是推荐商品评分和当前计算门店商品销量的spearman相关系数
    corr1_df=pd.merge(goods_score_df,goods_sales_amt_one_shop,on='goods_id')
    corr1_df['sales_amt']=pd.to_numeric(corr1_df['sales_amt'])
    corr1_result=corr1_df[['score','sales_amt']].corr(method='spearman')
    corr1=corr1_result.iloc[0,1]


    # 第二个相关性是当前计算门店销量和所有门店商品销量的spearman相关系数
    corr2_df_pre=pd.merge(goods_score_df,goods_sales_amt_one_shop,on='goods_id')
    corr2_df=pd.merge(corr2_df_pre,goods_sales_amt_all,on='goods_id')
    corr2_df['sales_amt']=pd.to_numeric(corr2_df['sales_amt'])
    corr2_df['sum(sales_amt)']=pd.to_numeric(corr2_df['sum(sales_amt)'])
    corr2_df_cal=corr2_df[['sales_amt','sum(sales_amt)']]
    corr2_result=corr2_df_cal.corr(method='spearman')
    corr2=corr2_result.iloc[0,1]

    # 把第一个、第二个相关性加入到记录结果的df中
    corr_record.loc[count]=[shop_name_pre,corr1,corr2]
    print(j,",",shop_name_pre,",",corr1,",",corr2)
    count+=1

