import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 해당 데이터는 kaggle에서 가져온 데이터이고 synthetically 하다고 명시 되어 있었음 (rink = https://www.kaggle.com/datasets/sharmajicoder/amazon-e-commerce)
amazon_db = pd.read_csv("/users/ihyeonho/Py_project/Amazon_e_commerce/data/amazon_e_commerce.csv") # 맥북용
#amazon_db = pd.read_csv("/users/ihyeonho/Desktop/JadonLee0709/Amazon-e-commerce-cs-analysis/data/amazon_e_commerce.csv") # 맥미니용

# 1. 배송 지연과 반품의 관계 (배송이 지연된 경우 구매자의 반품률이 증가하는가?)
# -> 가설: 배송이 오래 걸릴수록 고객 불만이 커져 반품으로 이어질 것이다
# -> 검증: 반품 그룹 vs 비반품 그룹의 평균 배송일을 t-test로 비교
# -> 다음: 상품 평점(rating), 판매자 평점(seller_rating)도 같은 방식으로 반품과의 관계 검증 예정

print("1. 배송 지연과 반품의 관계")
print()
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
def compare_by_return(df, col): # # is_returned 기준으로 그룹 나누고, col 값으로 비교 (배송기간, 평점 등)
    returned = df[df['is_returned'] == True][col]
    not_returned = df[df['is_returned'] == False][col]

    levene_result = stats.levene(returned, not_returned)
    ttest_result = stats.ttest_ind(returned, not_returned, equal_var =  (levene_result.pvalue > 0.05))
    # Welch's t-test 로 적용 p-value = 0이므로 p-value < 0.05 이하임

    print('[{}] 반품 평균: {:.4f}, 비반품 평균: {:.4f}'.format(col, returned.mean(), not_returned.mean()))
    print('Levene p-value: {:.4f}'.format(levene_result.pvalue))
    print('t-test p-value: {:.4f}'.format(ttest_result.pvalue))

    if ttest_result.pvalue < 0.05:
        print('→ 결론: {}와 반품 사이에는 통계적으로 유의미한 관계가 있음'.format(col))
    else:
        print('→ 결론: {}와 반품 사이에는 통계적으로 유의미한 차이가 없음'.format(col))

compare_by_return(amazon_db, 'shipping_time_days')
print()

# 2. 상품 평점과 반품의 관계 (상품 평점이 낮으면 반품률이 높은가?)
print("2. 상품 평점과 반품의 관계")
print()
sns.boxplot(x = 'is_returned', y = 'rating', data=amazon_db)
plt.xlabel('Is Returned')
plt.ylabel('Product Rating')
plt.title('Product Rating Distribution by Return Status ')

amazon_db_is_returned_rating_group_count = amazon_db.groupby('is_returned')['rating'].count()

plt.ylim(amazon_db['rating'].min() - 0.3, amazon_db['rating'].max() + 0.5)
for i, (group, count) in enumerate(amazon_db_is_returned_rating_group_count.items()):
    plt.text(i, amazon_db['rating'].max() + 0.05, 'n={}'.format(count), ha = 'center')


plt.show()

def count_outliers(series):
    q1 = series.quantile(0.25) # 1 사분위수 (박스 아래부분)
    q3 = series.quantile(0.75) # 3 사분위수 (박스 위 부분)
    iqr = q3 - q1 # 박스의 높
    lower_bound = q1 - 1.5 * iqr # 하한
    upper_bound = q3 + 1.5 * iqr # 상한
    outliers = series[(series < lower_bound) | (series > upper_bound)] # 이상치 -> 상한보다 크거나 하한보다 작은경우
    return len(outliers)

false_group = amazon_db[amazon_db['is_returned'] == False]['rating']
true_group = amazon_db[amazon_db['is_returned'] == True]['rating']

compare_by_return(amazon_db, 'rating') # 1번에 있던 함수 불러와서 결과 도출(반품에 따른 평점 분석)

print('False 그룹 이상치 개수:', count_outliers(false_group)) # 반품 안한 그룹
print('True 그룹 이상치 개수:', count_outliers(true_group)) # 반품 한 그룹
print()

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