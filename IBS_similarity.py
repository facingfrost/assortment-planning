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

"""-------------------- 按照用户画像数据计算相似度并进行推荐 --------------------"""
# 先读取所有需要的固定数据
# 先转化成UTF-8编码，读入数据之后按照MultiIndex的格式查看
user_profile=pd.read_table("用户画像精简版-8.txt",index_col=[0,1,2,3])

# 读取每个门店对每种商品的评分
query=r"SELECT shop_id,goods_id,sum(ratio) over(PARTITION BY shop_id ORDER BY rownum ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM shushi_tmp.wanglinhan_shop_goods_ratio_order_201911"
shop_goods_rating=get_data_from_hive(query)
shop_goods_rating.columns=['shop_id','goods_id','rating']

# 获取腾讯数据中出现的门店列表
query="SELECT distinct(shop_id) from shushi_tmp.bh_txun_new_amt_qty"
shop_list_df=get_data_from_hive(query)
# df格式转化为list
shop_list=shop_list_df.iloc[:,0].tolist()

# 取出用于检验推荐结果的数据,取屏西店2019年12月数据
query=r"SELECT goodsid,sum(sales_amt) from shushi.sale_sap_order_fct_new_shushi WHERE shop_id = '9010' and calday >= '20191201' and calday<='20191230' GROUP BY goodsid"
goods_amt=get_data_from_hive(query)

# 获取所有商品列表
query=r"SELECT DISTINCT(goods_id) from shushi_tmp.wanglinhan_shop_goods_ratio_201911"
goods_df_all=get_data_from_hive(query)
goods_list_all=goods_df_all['goods_id'].tolist()


# 计算门店两两之间的余弦相似度
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def get_sim_matrix(variables):
    """
    :param variables: 用于计算相似度的变量
    :return: 门店两两之间的相似度矩阵
    """
    sim_matrix_df=pd.DataFrame(columns=shop_list,index=shop_list)
    for i in range(len(shop_list)):
        for j in range(i,len(shop_list)):
            shop_1=shop_list[i]
            shop_2=shop_list[j]
            TGI_df_1=user_profile.loc[(shop_1,["工作","居住"],variables)]
            TGI_df_2=user_profile.loc[(shop_2,["工作","居住"],variables)]
            TGI_intersection=TGI_df_1.merge(TGI_df_2,on=["人员类型","标签名称","标签值"])
            TGI_list_1=TGI_intersection.loc[:,'TGI_x'].tolist()
            TGI_list_2=TGI_intersection.loc[:,'TGI_y'].tolist()
            np_TGI_list_1=np.array(TGI_list_1).reshape(1,-1)
            np_TGI_list_2=np.array(TGI_list_2).reshape(1,-1)
            sim_np=cosine_similarity(np_TGI_list_1,np_TGI_list_2)
            sim_list=sim_np.tolist()
            sim=sim_list[0][0]
            sim_matrix_df.loc[shop_1,shop_2]=sim
            sim_matrix_df.loc[shop_2,shop_1]=sim
    return sim_matrix_df


# 根据相似度进行跨店推荐
def recommendation_between_shops(shop_id,sim_matrix_df):
    """
    :param shop_id: 需要推荐的目标门店
    :param sim_matrix_df: 计算得到的相似度矩阵
    :return: 推荐商品列表，其中的元素是由推荐商品和推荐商品评分构成的元组
    """
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



"""-------------------- 计算auc的值 --------------------"""
sales_amt_sum=goods_amt['sum(sales_amt)'].sum() # 用于检验的数据的销量之和
goods_amt_dict=goods_amt.set_index('goodsid').T.to_dict('list') # 商品及其销量转换成字典形式

# 得到用于计算auc的列表
def get_auc_array(recommendation_goods):
    """
    :param recommendation_goods: 输入得到的推荐结果
    :return: 一个列表，用这个列表的数据计算auc
    """
    auc_y_pre=[]
    for i in recommendation_goods:
        if i in goods_amt_dict.keys():
            auc_y_pre.append(goods_amt_dict[i][0]/sales_amt_sum)
        else:
            auc_y_pre.append(0)
    auc_y=np.array(auc_y_pre).cumsum()
    return auc_y


# 计算AUC的值
def auc(x,y):
    """
    :param x: 推荐商品的个数
    :param y: 计算得到的auc_y
    :return: auc的值
    """
    s=0
    slide=1/x
    for i in range(x-1):
        s+=(float(y[i])+float(y[i+1]))/2*slide
    return s


# 打通计算全过程的函数
def direct_auc(shop_id,variables):
    sim_matrix_df_now=get_sim_matrix(variables) #根据指定的变量计算门店相似度
    recommendation_goods=recommendation_between_shops(shop_id,sim_matrix_df_now) # 获取推荐商品的列表
    auc_y=get_auc_array(recommendation_goods) # 获取计算auc的np.array
    return auc(6818,auc_y) #得到最终的auc值


"""---------- 通过逐步回归选择用于计算相似度的指标 --------------------"""
# 一共有APP偏好、人生阶段、出国游、到访偏好、婚否、子女年龄、学历、居住社区房价等级、居住社区房价（分段）、差旅常客、年龄（分段）、性别、
# 所在行业、手机价格（分段）、手机品牌、旅游距离、是否有车、消费水平（分段）、职业细分、职住距离（分段）、通勤方式、酒店消费价格等级、
# 酒店消费水平、餐饮消费水平这24个变量
# 目前采用向后选择法
feature_list=["APP偏好","人生阶段","出国游","到访偏好","婚否","子女年龄","学历","居住社区房价等级","居住社区房价（分段）","差旅常客","年龄（分段）","性别",
             "所在行业","手机价格（分段）","手机品牌","旅游距离","是否有车","消费水平（分段）","职业细分","职住距离（分段）","通勤方式","酒店消费价格等级",
              "酒店消费水平","餐饮消费水平"]

# 减少每一个变量，看计算出来的auc能否被优化
def check_variables(shop_id,feature_list,auc_pre):
    """
    :param shop_id: 当前计算的门店id
    :param feature_list: 当前选择的特征列表
    :param auc_pre: 当前的auc
    :return: 下一步计算得到的最大auc，下一步计算得到的特征列表，上一步去掉的变量索引
    """
    auc_list=[]
    auc_max=auc_pre
    out_variable_index=-1
    feature_list_now=[]
    for i in range(len(feature_list)):
        flag_list=[1 for i in range(len(feature_list))] # 表示是否选取这个变量的列表
        feature_list_pre=[] # 要计算auc的列表
        flag_list[i]=0 # 依次删除的变量
        for j in range(len(feature_list)):
            if flag_list[j]==1:
                feature_list_pre.append(feature_list[j]) # 得到现在要计算的特征列表
        auc_process=direct_auc(shop_id,feature_list_pre) # 计算当前特征列表下的auc
        if auc_process > auc_max:
            out_variable_index=i #记录要去除的变量
            auc_max=auc_process #记录现在auc的最大值
            feature_list_now=feature_list_pre
        print(i,":",feature_list_pre)
        print(auc_process)
    print("选择的特征是：",feature_list_now)
    print("此时的auc是",auc_max)
    return auc_max,feature_list_now,out_variable_index


# 逐步回归的主函数
feature_list_now=feature_list
auc_max_now=auc_pre
i=1
while len(feature_list_now):
    print("第",i,"次检查向后检查变量")
    auc_max_now,feature_list_now,out_variable_index=check_variables('9010',feature_list_now,auc_max_now) #输入初始值进行判断
    if out_variable_index==-1:
        print("最终选择的变量是：",feature_list_now)
        print("最大的auc是：",auc_max_now)
        break
    else:
        i+=1


# 最终选择的变量是[居住社区房价（分段），职住距离（分段），通勤方式，酒店消费价格等级，餐饮消费水平]

