import pandas as pd
from pandas import col
from scipy import stats
import matplotlib.pyplot as plt

# 해당 데이터는 kaggle에서 가져온 데이터이고 synthetically 하다고 명시 되어 있었음 (rink = https://www.kaggle.com/datasets/sharmajicoder/amazon-e-commerce)
amazon_db = pd.read_csv("/users/ihyeonho/Py_project/Amazon_e_commerce/data/amazon_e_commerce.csv") # 맥북용
#amazon_db = pd.read_csv("/users/ihyeonho/Desktop/JadonLee0709/Amazon-e-commerce-cs-analysis/data/amazon_e_commerce.csv") # 맥미니용

# 1. 배송 지연과 반품의 관계 (배송이 지연된 경우 구매자의 반품률이 증가하는가?)
# -> 가설: 배송이 오래 걸릴수록 고객 불만이 커져 반품으로 이어질 것이다
# -> 검증: 반품 그룹 vs 비반품 그룹의 평균 배송일을 t-test로 비교
# -> 다음: 상품 평점(rating), 판매자 평점(seller_rating)도 같은 방식으로 반품과의 관계 검증 예정

amazon_db_ship_return_group = amazon_db.groupby('shipping_time_days')['is_returned'].mean()

# ** 차트 생성 파트
(amazon_db_ship_return_group * 100).plot(kind='bar')
plt.xlabel('Shipping Time Days')
plt.ylabel('Return Rate (%)')
plt.title('Return Rate by Shipping Time Days')
plt.xticks(rotation = 0)

for i, value in enumerate(amazon_db_ship_return_group * 100):
    plt.text(i, value + 0.3, '{:.1f}%'.format(value), ha = 'center')

plt.show()


# ** 배송 지연과 반품의 관계 t-test 검정, levene 검정
# -> t-test 전에 Levene으로 두 그룹의 분산이 같은지 먼저 확인 (등분산 가정 여부에 따라 t-test 계산 방식이 달라지기 때문)
def compare_by_return(df, col):
    returned = df[df['is_returned'] == True][col]
    not_returned = df[df['is_returned'] == False][col]

    levene_result = stats.levene(returned, not_returned)
    ttest_result = stats.ttest_ind(returned, not_returned, equal_var =  (levene_result.pvalue > 0.05))
    # Welch's t-test 로 적용 p-value = 0이므로 p-value < 0.05 이하임

    print('[{}] 반품 평균: {:.4f}, 비반품 평균: {:.4f}'.format(col, returned.mean(), not_returned.mean()))
    print('Levene p-value: {:.4f}'.format(levene_result.pvalue))
    print('t-test p-value: {:.4f}'.format(ttest_result.pvalue))

    if ttest_result.pvalue < 0.05:
        print('→ 결론: 배송 지연과 반품 사이에는 통계적으로 유의미한 관계가 있음')
    else:
        print('→ 결론: 통계적으로 유의미한 차이가 없음')

compare_by_return(amazon_db, 'shipping_time_days')

# 2. 상품 평점과 반품의 관계 (상품 평점이 낮으면 반품률이 높은가?)


# 3. 판매자의 평점과 반품의 관계 (판매자의 평점이 낮을수록, 반품률이 증가하는가?)



# 재구매 횟수 계산 (user_id + seller_id 조합별 구매 횟수)
#repurchase_count = amazon_db.groupby(['user_id', 'seller_id']).size().reset_index(name='purchase_count')

# print(amazon_db['user_id'].nunique())          # 고유 유저 수
# print(amazon_db['seller_id'].nunique())        # 고유 판매자 수
# print(repurchase_count['purchase_count'].value_counts())  # 재구매 횟수 분포
#
# print(pd.crosstab(amazon_db['delivery_status'], amazon_db['is_returned']))

# print(amazon_db['shipping_time_days'].describe())
# print(amazon_db['shipping_time_days'].value_counts().sort_index())