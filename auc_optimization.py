"""--------------------- 配置环境和所需函数 --------------------"""

from pyhive import hive
import pandas as pd
import datetime

import datetime

starttime = datetime.datetime.now()

#连接云超的数据库,端口号，hive:10001,impala:21051
def get_data_from_hive(query):
    conn=hive.connect(host='10.1.53.19',port=21051,username='wangrui',password='95rPZEytgAnuxCw',auth='LDAP')
    cur=conn.cursor()
    cur.execute(query)
    data=cur.fetchall()
    columnDes=cur.description #获取连接对象的描述信息
    columnNames=[columnDes[i][0] for i in range(len(columnDes))]
    df=pd.DataFrame([list(i) for i in data],columns=columnNames)
    cur.close()
    return df

"""-------------------- 读取数据用于推荐模型的训练 --------------------"""

query='''with t1 as (
    SELECT shop_id, goodsid, sum(sales_qty) as sales
        from shushi.sale_sap_order_fct_new_shushi 
        WHERE calday >= "20191101" and calday <= "20191131"
        and shop_id in ('9134',  
                   '9660','9029','9629','9144','9872','9M4R','9L34','9039','9020','9074','9041','9017','9531','9L39',
                   '9700','9015','9296','9L12','9439','9019','9010','9L10','9064','9L33','9M4N','9L57','90T3','9018',
                   '9078','9459','9013','9068','9L36','9496','9L28','9438','9030','9839','95C8','9047','9025','9139',
                   '9387','9116','9165','9L40','9046','90K5','9L23','9023','9893','9701','9033','9M4Q','9079','9L64',
                   '9L13','9L59','9007','9011','9014','9L54','9285','9L61','9L56','9225','9163','9330','9187','9559',
                   '9009','90S7','9105','9129','9712','90U7','9038','9113','9440','9067','90V3','9L11','9199','9549',
                   '9024','90S8','9045','9120','9L07','9012','9049','9026','9M4V','9M0K','9M5R')
        group by shop_id, goodsid
    ),
     t2 as (
    select shop_id, goodsid, sales, row_number() over(partition by shop_id order by sales desc) as sales_rank
        from t1
    )
select shop_id, goodsid, sales from t2 where sales_rank <= 300 order by shop_id, sales_rank
'''
shop_goods_sales_amt=get_data_from_hive(query)

# 以csv的形式保存到本地
shop_goods_sales_amt.to_csv("shop_goods_sales_qty_300.csv",index=False)


# 以multiindex的形式读取本地csv
shop_goods_sales_amt=pd.read_csv("shop_goods_sales_qty_300.csv",index_col=[0,1])


# 读取每个门店对每种商品的评分
query=r"SELECT shop_id,goods_id,sum(ratio) over(PARTITION BY shop_id ORDER BY rownum ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM shushi_tmp.wanglinhan_shop_goods_ratio_order_201911"
shop_goods_rating=get_data_from_hive(query)
shop_goods_rating.columns=['shop_id','goods_id','rating']


# 获取腾讯数据中出现的门店列表
query="SELECT distinct(shop_id) from shushi_tmp.bh_txun_new_amt_qty"
shop_list_df=get_data_from_hive(query)
# df格式转化为list
shop_list=shop_list_df.iloc[:,0].tolist()



# 获取所有商品列表
query=r"SELECT DISTINCT(goods_id) from shushi_tmp.wanglinhan_shop_goods_ratio_201911"
goods_df_all=get_data_from_hive(query)
goods_list_all=goods_df_all['goods_id'].tolist()


# 取出用于检验推荐结果的数据,取屏西店2019年12月数据
query=r"SELECT goodsid,sum(sales_amt) from shushi.sale_sap_order_fct_new_shushi WHERE shop_id = '9010' and calday >= '20191201' and calday<='20191230' GROUP BY goodsid"
goods_amt=get_data_from_hive(query)


"""-------------------- 计算相似度 --------------------"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
sim_matrix_df=pd.DataFrame(columns=shop_list,index=shop_list)
for i in range(len(shop_list)):
    for j in range(i,len(shop_list)):
        shop_1=shop_list[i]
        shop_2=shop_list[j]
        shop_goods_sales_amt_df_1=shop_goods_sales_amt.loc[(shop_1,slice(None)),:]
        shop_goods_sales_amt_df_2=shop_goods_sales_amt.loc[(shop_2,slice(None)),:]
        shop_goods_sales_amt_df_intersection=shop_goods_sales_amt_df_1.merge(shop_goods_sales_amt_df_2,on=['goodsid'])
        sales_list_1=shop_goods_sales_amt_df_intersection.loc[:,'sales_x'].tolist()
        sales_list_2=shop_goods_sales_amt_df_intersection.loc[:,'sales_y'].tolist()
        np_sales_list_1=np.array(sales_list_1).reshape(1,-1)
        np_sales_list_2=np.array(sales_list_2).reshape(1,-1)
        sim_np=cosine_similarity(np_sales_list_1,np_sales_list_2)
        sim_list=sim_np.tolist()
        sim=sim_list[0][0]
        len1=len(shop_goods_sales_amt_df_1)
        len2=len(shop_goods_sales_amt_df_2)
        sim_type=len(shop_goods_sales_amt_df_intersection)/sqrt(len1*len2)
        sim_overall=sim*sim_type
        sim_matrix_df.loc[shop_1,shop_2]=sim_overall
        sim_matrix_df.loc[shop_2,shop_1]=sim_overall


"""-------------------- 选择排名前20的门店进行推荐 --------------------"""
# 取出9010门店的相似度series并倒序排列
sim_series=sim_matrix_df['9010'].sort_values(ascending=False)

# 取出与9010门店相似度在前20的门店
non_zero_list=list(sim_series.index)[0:60]

# 其余的门店相似度都改为0
for i in shop_list:
    if i in non_zero_list:
        continue
    else:
        sim_matrix_df.loc['9010',i]=0


"""-------------------- 进行跨店推荐 --------------------"""
def recommendation_between_shops(shop_id,sim_matrix_df):
    zero_list=[]
    for i in range(len(goods_list_all)):
        zero_list.append(0)
    goods_score_dict=dict(zip(goods_list_all,zero_list)) # 创建空白商品评分字典
    for i in range(len(shop_goods_rating)):
        shop_1=shop_goods_rating.iloc[i,0]
        if shop_1 != shop_id and shop_1 in shop_list:
            goods_score_dict[shop_goods_rating.iloc[i,1]]+=shop_goods_rating.iloc[i,2]*sim_matrix_df.loc[shop_id,shop_1]#商品评分*商店相似度
    recommendation_info=sorted(goods_score_dict.items(),key=lambda item:item[1],reverse=True)
    recommendation_goods=[]
    for i in range(6818):
        recommendation_goods.append(recommendation_info[i][0])
    return recommendation_goods


sales_amt_sum=goods_amt['sum(sales_amt)'].sum()
goods_amt_dict=goods_amt.set_index('goodsid').T.to_dict('list')


def get_auc_array(recommendation_goods):
    auc_y_pre=[]
    for i in recommendation_goods:
        if i in goods_amt_dict.keys():
            auc_y_pre.append(goods_amt_dict[i][0]/sales_amt_sum)
        else:
            auc_y_pre.append(0)
    auc_y=np.array(auc_y_pre).cumsum()
    return auc_y


# 计算AUC的值
def auc(x,y): #x是推荐商品数量，y是计算出来的auc_y
    s=0
    slide=1/x
    for i in range(x-1):
        s+=(float(y[i])+float(y[i+1]))/2*slide
    return s


"""-------------------- 调用函数运行 -------------------"""
recommendation_goods=recommendation_between_shops('9010',sim_matrix_df)
auc_y=get_auc_array(recommendation_goods)
auc(6818,auc_y)




"""-------------------- 不计算相似度运行一个比较的baseline --------------------"""


# 在商品评分大于0.37的商品中，随机选取6818种商品，并按商品评分倒序排列
shop_goods_rating_baseline=shop_goods_rating[shop_goods_rating['rating']>=0.37]
goods_list_baseline=list(set(shop_goods_rating_baseline['goods_id']))
from random import sample
recommendation_goods=sample(goods_list_baseline,6818)
recommendation_goods.reverse()

# 被推荐的商品在商店中的评分相加并倒序排列，得到要推荐的商品
shop_goods_rating_chosen=shop_goods_rating[shop_goods_rating['goods_id'].isin(recommendation_goods)]
shop_goods_rating_chosen = shop_goods_rating_chosen.drop(shop_goods_rating_chosen[shop_goods_rating_chosen['shop_id'] == '9010'].index)
shop_goods_rating_chosen=shop_goods_rating_chosen.drop(['shop_id'],axis=1)
shop_goods_rating_chosen_sum=shop_goods_rating_chosen.groupby('goods_id').sum()
shop_goods_rating_chosen_sum_sort=shop_goods_rating_chosen_sum.sort_values(['rating'],ascending=False)
recommendation_goods=list(shop_goods_rating_chosen_sum_sort.index)

# 计算auc
auc_y=get_auc_array(recommendation_goods)
auc(6818,auc_y)




