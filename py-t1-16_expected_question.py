# 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고
# 앞에서 부터 20개의 데이터를 추출한 후 
# f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)

# - 데이터셋 : basic1.csv 
# - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# - File -> Editor Type -> Script

import numpy as np
import pandas as pd
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(df)

# 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고
df_f2_0 = df[df.f2 == 0].sort_values(by = 'age').reset_index(drop = True)

# 앞에서 부터 20개의 데이터를 추출한 후 
df_f2_0_20 = df_f2_0.iloc[0:20,]

# f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)
print(df_f2_0_20.f1.fillna(df_f2_0_20.f1.min()).var())
print(df_f2_0_20.f1.var())
print(round(abs(df_f2_0_20.f1.fillna(df_f2_0_20.f1.min()).var() - df_f2_0_20.f1.var()),2))
# 정답 : 38.44

# kaggle answer

import pandas as pd

# 데이터 불러오기
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")

# f2가 0인 데이터 정렬(age 오름차순)
cond = (df['f2']==0)
df = df[cond].sort_values('age', ascending=True).reset_index(drop=True)

# 앞에서 부터 20개의 데이터 
df = df[:20]

# f1 결측치(최소값)를 채우기 전과 후의 분산
df_var1 = df['f1'].var()
df['f1'] = df['f1'].fillna(df['f1'].min())
df_var2 = df['f1'].var()

# 소수점 둘째자리까지 출력
print(round(df_var1 - df_var2, 2))

# 정답 : 38.44
