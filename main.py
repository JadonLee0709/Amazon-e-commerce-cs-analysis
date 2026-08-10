import pandas as pd
from scipy import stats


# 해당 데이터는 kaggle에서 가져온 데이터이고 synthetically 하다고 명시 되어 있었음 (rink = https://www.kaggle.com/datasets/sharmajicoder/amazon-e-commerce)
#amazon_db = pd.read_csv("/users/ihyeonho/Py_project/Amazon_e_commerce/data/amazon_e_commerce.csv") # 맥북용
amazon_db = pd.read_csv("/users/ihyeonho/Desktop/JadonLee0709/Amazon-e-commerce-cs-analysis/data/amazon_e_commerce.csv") # 맥미니용

# 1. 배송 지연과 반품의 관계 (배송이 지연된 경우 구매자의 반품률이 증가하는가?)


# 2. 상품 평점과 반품의 관계 (상품 평점이 낮으면 반품률이 높은가?)


# 3. 판매자의 평점과 반품의 관계 (판매자의 평점이 낮을수록, 반품률이 증가하는가?)



# 재구매 횟수 계산 (user_id + seller_id 조합별 구매 횟수)
repurchase_count = amazon_db.groupby(['user_id', 'seller_id']).size().reset_index(name='purchase_count')

# print(amazon_db['user_id'].nunique())          # 고유 유저 수
# print(amazon_db['seller_id'].nunique())        # 고유 판매자 수
# print(repurchase_count['purchase_count'].value_counts())  # 재구매 횟수 분포
#
# print(pd.crosstab(amazon_db['delivery_status'], amazon_db['is_returned']))

print(amazon_db['shipping_time_days'].describe())
print(amazon_db['shipping_time_days'].value_counts().sort_index())