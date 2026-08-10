import pandas as pd

amazon_db = pd.read_csv("/users/ihyeonho/Py_project/Amazon_e_commerce/data/amazon_e_commerce.csv")

# 1. 배송 지연과 반품의 관계 (배송이 지연된 경우 구매자의 반품률이 증가하는가?)


# 2. 상품 평점과 반품의 관계 (상품 평점이 낮으면 반품률이 높은가?)


# 3. 판매자의 평점과 반품의 관계 (판매자의 평점이 낮을수록, 반품률이 증가하는가?)



# 재구매 횟수 계산 (user_id + seller_id 조합별 구매 횟수)
repurchase_count = amazon_db.groupby(['user_id', 'seller_id']).size().reset_index(name='purchase_count')

print(amazon_db['user_id'].nunique())          # 고유 유저 수
print(amazon_db['seller_id'].nunique())        # 고유 판매자 수
print(repurchase_count['purchase_count'].value_counts())  # 재구매 횟수 분포