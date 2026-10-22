import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

# 해당 데이터는 kaggle에서 가져온 데이터이고 synthetically 하다고 명시 되어 있었음 (link = https://www.kaggle.com/datasets/sharmajicoder/amazon-e-commerce)
amazon_db = pd.read_csv("/users/ihyeonho/Py_project/Amazon_e_commerce/data/amazon_e_commerce.csv") # 맥북용
#amazon_db = pd.read_csv("/users/ihyeonho/Desktop/JadonLee0709/Amazon-e-commerce-cs-analysis/data/amazon_e_commerce.csv") # 맥미니용

# 1. 배송 기간과 반품의 관계 (배송 기간이 길수록 구매자의 반품률이 증가하는가?)
# -> 가설: 배송이 오래 걸릴수록 고객 불만이 커져 반품으로 이어질 것이다
# -> 검증: 반품 그룹 vs 비반품 그룹의 평균 배송일을 t-test로 비교
# -> 다음: 상품 평점(rating), 판매자 평점(seller_rating)도 같은 방식으로 반품과의 관계 검증 예정

print("0. 데이터 검증: delivery_status vs is_returned") # readme에 결과 기록하고 검증의미 없는거 알리기 (지연과 반품관계 설명 x, 그러나 뒤에 배송기간과 반품으로 설명)
print()
print(pd.crosstab(amazon_db['delivery_status'], amazon_db['is_returned']))
print()

print("1. 배송 기간과 반품의 관계")
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


# ** 배송 기간과 반품의 관계 t-test 검정, levene 검정
# -> t-test 전에 Levene으로 두 그룹의 분산이 같은지 먼저 확인 (등분산 가정 여부에 따라 t-test 계산 방식이 달라지기 때문)
def compare_by_return(df, col): # # is_returned 기준으로 그룹 나누고, col 값으로 비교 (배송기간, 평점 등) p-value 연산식
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

    print('반품 분산: {:.4f}, 비반품 분산: {:.4f}'.format(returned.var(), not_returned.var())) # 반품 비반품 분산

compare_by_return(amazon_db, 'shipping_time_days')
print()

# 2. 상품 평점과 반품의 관계 (상품 평점이 낮으면 반품률이 높은가?)
print("2. 상품 평점과 반품의 관계")
print()
sns.boxplot(x = 'is_returned', y = 'rating', data=amazon_db)
plt.xlabel('Is Returned')
plt.ylabel('Product Rating')
plt.title('Product Rating Distribution by Return Status')

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

rating_false_group = amazon_db[amazon_db['is_returned'] == False]['rating']
rating_true_group = amazon_db[amazon_db['is_returned'] == True]['rating']

compare_by_return(amazon_db, 'rating') # 1번에 있던 함수 불러와서 결과 도출(반품에 따른 평점 분석)

print('False 그룹 이상치 개수:', count_outliers(rating_false_group)) # 반품 안한 그룹
print('True 그룹 이상치 개수:', count_outliers(rating_true_group)) # 반품 한 그룹
print()

# 평점 구간별 반품률 (평균 차이는 작아도 낮은 평점 구간에 반품이 몰려 있는지 확인)
rating_floor_group = amazon_db.groupby(np.floor(amazon_db['rating']))['is_returned'].agg(['mean', 'size'])
rating_floor_group['mean'] = rating_floor_group['mean'] * 100
rating_floor_group.columns = ['return_rate(%)', 'orders']
print('평점 구간별 반품률 (소수점 버림 기준, 예: 3 = 3.0~3.9점):')
print(rating_floor_group)
print()

# 3. 판매자의 평점과 반품의 관계 (판매자의 평점이 낮을수록, 반품률이 증가하는가?)
print("3. 판매자의 평점과 반품의 관계")
print()
sns.boxplot(x = 'is_returned', y = 'seller_rating', data=amazon_db)
plt.xlabel('Is Returned')
plt.ylabel('Seller Rating')
plt.title('Seller Rating Distribution by Return Status')

amazon_db_is_returned_seller_rating_group_count = amazon_db.groupby('is_returned')['seller_rating'].count()

plt.ylim(amazon_db['seller_rating'].min() - 0.3, amazon_db['seller_rating'].max() + 0.5)
for i, (group, count) in enumerate(amazon_db_is_returned_seller_rating_group_count.items()):
    plt.text(i, amazon_db['seller_rating'].max() + 0.05, 'n={}'.format(count), ha = 'center')

plt.show()

seller_rating_false_group = amazon_db[amazon_db['is_returned'] == False]['seller_rating']
seller_rating_true_group = amazon_db[amazon_db['is_returned'] == True]['seller_rating']

compare_by_return(amazon_db, 'seller_rating')

print('Seller Rating False 그룹 이상치 개수:', count_outliers(seller_rating_false_group)) # 반품 안한 그룹
print('Seller Rating True 그룹 이상치 개수:', count_outliers(seller_rating_true_group)) # 반품 한 그룹
print()

# 4. 로지스틱 회귀 - 반품에 영향을 미치는 요인 종합 분석
# -> 가설: 배송일, 상품 평점, 판매자 평점, 할인율을 동시에 고려했을 때 각 변수가 반품에 미치는 영향은?
# -> 방법: 로지스틱 회귀 (종속변수: is_returned, 독립변수: 4개)
# -> delivery_status는 is_returned와 완전히 겹치는 열이므로(0번 검증) 독립변수에서 제외

print("4. 로지스틱 회귀 - 반품 영향 요인 종합 분석")
print()

X = amazon_db[['shipping_time_days', 'rating', 'seller_rating', 'discount']]
y = amazon_db['is_returned'].astype(int)

X = sm.add_constant(X)

model = sm.Logit(y, X)
result = model.fit()

# 오즈비(Odds Ratio) 계산 및 해석
# -> 오즈비는 확률이 아니라 오즈(반품/비반품 비율)의 배수
print()
print('=== 오즈비(Odds Ratio) 해석 ===')
odds_ratios = np.exp(result.params)
p_values = result.pvalues

for var in ['shipping_time_days', 'rating', 'seller_rating', 'discount']:
    odds = odds_ratios[var]
    p = p_values[var]
    if p < 0.05:
        direction = '증가' if odds > 1 else '감소'
        print('{}: 오즈비 {:.4f} (p={:.4f}) → 1단위 증가 시 반품 오즈 {:.1f}% {}'.format(
            var, odds, p, abs((odds - 1) * 100), direction))
    else:
        print('{}: 오즈비 {:.4f} (p={:.4f}) → 통계적으로 유의미하지 않음'.format(var, odds, p))

print(result.summary())
print('McFadden Pseudo R²: {:.4f}'.format(result.prsquared))
print()

# 4-1. 로지스틱 회귀 보강
# (a) 6일(최장 배송) 더미 변수: 반품률이 1~5일은 비슷하다가 6일에서만 뛰는 계단형이라, 선형 항보다 더미가 실제 패턴에 맞음
# (b) 표준화(1 표준편차 기준) 오즈비: 변수마다 단위가 달라 원래 오즈비끼리는 크기 비교가 안 되므로 표준화 후 비교
print("4-1. 로지스틱 회귀 보강 (6일 더미 + 표준화)")
print()

max_ship = amazon_db['shipping_time_days'].max()
amazon_db['ship_longest'] = (amazon_db['shipping_time_days'] == max_ship).astype(int)

dummy_cols = ['ship_longest', 'rating', 'seller_rating', 'discount']
X_dummy = sm.add_constant(amazon_db[dummy_cols])
result_dummy = sm.Logit(y, X_dummy).fit(disp=0)

print('=== (a) 최장 배송({}일) 더미 포함 모델 오즈비 ==='.format(max_ship))
odds_dummy = np.exp(result_dummy.params)
for var in dummy_cols:
    print('{}: 오즈비 {:.4f} (p={:.4f})'.format(var, odds_dummy[var], result_dummy.pvalues[var]))
print('McFadden Pseudo R²: {:.4f} (선형 배송일 모델 {:.4f})'.format(result_dummy.prsquared, result.prsquared))
print()

std_cols = ['shipping_time_days', 'rating', 'seller_rating', 'discount']
X_std = (amazon_db[std_cols] - amazon_db[std_cols].mean()) / amazon_db[std_cols].std()
result_std = sm.Logit(y, sm.add_constant(X_std)).fit(disp=0)

print('=== (b) 표준화 오즈비 (1 표준편차 증가 기준, 변수끼리 비교 가능) ===')
odds_std = np.exp(result_std.params).drop('const')
print(odds_std.sort_values(ascending=False))
print()

# 5. 카테고리와 반품의 관계 (카테고리에 따라 반품률이 다른가?)
# -> 방법: 카이제곱 독립성 검정 (범주형 vs 범주형), 효과 크기는 Cramér's V
print("5. 카테고리와 반품의 관계")
print()

category_return_table = pd.crosstab(amazon_db['category'], amazon_db['is_returned'])
chi2, chi_p, dof, expected = stats.chi2_contingency(category_return_table)
n_total = category_return_table.values.sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(category_return_table.shape) - 1)))

print(category_return_table)
print('카이제곱 통계량: {:.4f}, 자유도: {}, p-value: {:.4f}'.format(chi2, dof, chi_p))
print("Cramér's V (효과 크기): {:.4f}".format(cramers_v)) # 0에 가까울수록 관계 없음 (0.1 미만이면 매우 약함)
print('기대빈도 최솟값: {:.1f} (5 이상이면 카이제곱 가정 충족)'.format(expected.min()))

if chi_p < 0.05:
    print('→ 결론: 카테고리와 반품 사이에는 통계적으로 유의미한 관계가 있음')
else:
    print('→ 결론: 카테고리와 반품 사이에는 통계적으로 유의미한 관계가 없음')

category_return_rate = amazon_db.groupby('category')['is_returned'].mean().sort_values(ascending=False) * 100
category_return_rate.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Return Rate (%)')
plt.title('Return Rate by Category')
plt.xticks(rotation = 45, ha = 'right')
for i, value in enumerate(category_return_rate):
    plt.text(i, value + 0.1, '{:.1f}%'.format(value), ha = 'center')
plt.tight_layout()
plt.show()
print()

# ----------------------------------------------------------------------
# 여기서부터는 판매량 분석 (SQL로 집계한 결과와 비교할 용도)
# -> 판매량 기준: 주문 건수 (행 1개 = 주문 1건). quantity 열이 없어서 건수로 정의함
# ----------------------------------------------------------------------

# 6. 브랜드별 판매량 (주제: 제조사별 판매량, manufacturer 열이 없어 brand로 대체)
print("6. 브랜드별 판매량")
print()
brand_sales = amazon_db['brand'].value_counts()
print('브랜드 개수:', len(brand_sales))
print(brand_sales.head(10))
print('1위 / 최하위 판매량 비율: {:.2f}배'.format(brand_sales.max() / brand_sales.min())) # 1에 가까우면 브랜드 간 차이가 거의 없음

brand_sales.head(10).plot(kind='bar')
plt.xlabel('Brand')
plt.ylabel('Orders')
plt.title('Top 10 Brands by Orders')
plt.xticks(rotation = 45, ha = 'right')
for i, value in enumerate(brand_sales.head(10)):
    plt.text(i, value, '{:,}'.format(value), ha = 'center', va = 'bottom')
plt.tight_layout()
plt.show()
print()

# 7. 월별 카테고리 판매량 (구매일에 따라 어떤 카테고리가 많이 팔리는가?)
# -> purchase_date가 text이므로 먼저 날짜로 변환 (형식이 맞지 않으면 NaT가 되므로 개수를 확인)
print("7. 월별 카테고리 판매량")
print()
amazon_db['purchase_date'] = pd.to_datetime(amazon_db['purchase_date'], errors = 'coerce')
print('날짜 변환 실패(NaT) 개수:', amazon_db['purchase_date'].isna().sum())
print('구매일 범위: {} ~ {}'.format(amazon_db['purchase_date'].min(), amazon_db['purchase_date'].max()))

amazon_db['year_month'] = amazon_db['purchase_date'].dt.to_period('M').astype(str)
monthly_category = amazon_db.groupby(['year_month', 'category']).size().unstack(fill_value = 0)
print(monthly_category.head())
print()
print('월별 판매량 1위 카테고리:')
print(monthly_category.idxmax(axis = 1))

# 월에 따라 카테고리 비중이 달라지는지 카이제곱 검정
# -> 주문이 너무 적은 달(데이터 시작일이 월 중간이라 며칠치만 있는 달)은 비교가 왜곡되므로 제외
monthly_total = monthly_category.sum(axis = 1)
partial_months = monthly_total[monthly_total < monthly_total.median() * 0.5].index.tolist()
print('검정에서 제외한 달(주문 수가 중앙값의 절반 미만):', partial_months)
month_category_test = monthly_category.drop(index = partial_months)
chi2_m, p_m, dof_m, _ = stats.chi2_contingency(month_category_test)
v_m = np.sqrt(chi2_m / (month_category_test.values.sum() * (min(month_category_test.shape) - 1)))
print('월 x 카테고리 카이제곱: {:.4f}, 자유도: {}, p-value: {:.4f}, Cramér\'s V: {:.4f}'.format(chi2_m, dof_m, p_m, v_m))
print('월별 주문 건수 범위(제외한 달 빼고): {:,} ~ {:,}'.format(month_category_test.sum(axis = 1).min(), month_category_test.sum(axis = 1).max()))

monthly_category.plot(figsize = (12, 6))
plt.xlabel('Year-Month')
plt.ylabel('Orders')
plt.title('Monthly Orders by Category')
plt.xticks(rotation = 45)
plt.legend(title = 'Category', bbox_to_anchor = (1.02, 1), loc = 'upper left')
plt.tight_layout()
plt.show()
print()

# 8. 판매자 평점 구간별 판매량 (판매자 평점에 따라 판매량이 달라지는가?)
# -> 평점을 같은 폭의 5개 구간으로 나눠서 구간별 주문 건수와 반품률을 같이 확인
print("8. 판매자 평점 구간별 판매량")
print()
seller_rating_bin = pd.cut(amazon_db['seller_rating'], bins = 5)
seller_rating_sales = amazon_db.groupby(seller_rating_bin, observed = True).agg(
    orders = ('is_returned', 'size'),
    return_rate = ('is_returned', 'mean')
)
seller_rating_sales['return_rate'] = seller_rating_sales['return_rate'] * 100
print(seller_rating_sales)

seller_rating_sales.index = seller_rating_sales.index.astype(str)
seller_rating_sales['orders'].plot(kind = 'bar')
plt.xlabel('Seller Rating Range')
plt.ylabel('Orders')
plt.title('Orders by Seller Rating Range')
plt.xticks(rotation = 0)
for i, value in enumerate(seller_rating_sales['orders']):
    plt.text(i, value, '{:,}'.format(value), ha = 'center', va = 'bottom')
plt.tight_layout()
plt.show()
print()

# 9. 지역별 브랜드 판매량 (지역마다 많이 팔리는 브랜드가 다른가?)
# -> 지역과 브랜드가 많으면 표가 너무 커지므로 판매량 상위 10개씩만 비교
print("9. 지역별 브랜드 판매량")
print()
print('지역 개수:', amazon_db['location'].nunique())

top_locations = amazon_db['location'].value_counts().head(10).index
top_brands = amazon_db['brand'].value_counts().head(10).index
location_brand_sub = amazon_db[amazon_db['location'].isin(top_locations) & amazon_db['brand'].isin(top_brands)]
location_brand_table = pd.crosstab(location_brand_sub['location'], location_brand_sub['brand'])

print('지역별 판매량 1위 브랜드:')
print(location_brand_table.idxmax(axis = 1))

# 지역과 브랜드가 독립인지 카이제곱 검정 (상위 10개가 아니라 전체 지역 x 전체 브랜드 기준)
location_brand_all = pd.crosstab(amazon_db['location'], amazon_db['brand'])
chi2_lb, p_lb, dof_lb, _ = stats.chi2_contingency(location_brand_all)
v_lb = np.sqrt(chi2_lb / (location_brand_all.values.sum() * (min(location_brand_all.shape) - 1)))
print('지역별 주문 건수:')
print(amazon_db['location'].value_counts())
print('지역 x 브랜드 카이제곱: {:.4f}, 자유도: {}, p-value: {:.4f}, Cramér\'s V: {:.4f}'.format(chi2_lb, dof_lb, p_lb, v_lb))

plt.figure(figsize = (12, 7))
sns.heatmap(location_brand_table, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Brand')
plt.ylabel('Location')
plt.title('Orders by Location and Brand (Top 10 each)')
plt.tight_layout()
plt.show()
print()

# 10. 카테고리별 평균 할인율과 판매량 (할인율이 높은 카테고리가 더 많이 팔리는가?)
# -> discount 단위(0.15인지 15인지)를 먼저 확인
print("10. 카테고리별 평균 할인율과 판매량")
print()
print('discount 분포:')
print(amazon_db['discount'].describe())
print()

category_discount_sales = amazon_db.groupby('category').agg(
    avg_discount = ('discount', 'mean'),
    orders = ('is_returned', 'size'),
    return_rate = ('is_returned', 'mean')
)
category_discount_sales['return_rate'] = category_discount_sales['return_rate'] * 100
print(category_discount_sales.sort_values('avg_discount', ascending = False))

pearson_r, pearson_p = stats.pearsonr(category_discount_sales['avg_discount'], category_discount_sales['orders'])
spearman_r, spearman_p = stats.spearmanr(category_discount_sales['avg_discount'], category_discount_sales['orders'])
print('카테고리 개수(표본 수): {}'.format(len(category_discount_sales))) # 표본이 적으면 상관계수 신뢰도가 낮음
print('Pearson 상관계수: {:.4f} (p={:.4f})'.format(pearson_r, pearson_p))
print('Spearman 상관계수: {:.4f} (p={:.4f})'.format(spearman_r, spearman_p))

plt.scatter(category_discount_sales['avg_discount'], category_discount_sales['orders'])
for category, row in category_discount_sales.iterrows():
    plt.text(row['avg_discount'], row['orders'], category, fontsize = 8)
plt.xlabel('Average Discount')
plt.ylabel('Orders')
plt.title('Average Discount vs Orders by Category')
plt.tight_layout()
plt.show()
