# city와 f4를 기준으로 f5의 평균값을 구한 다음, f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)
# - 데이터셋 : basic1.csv 
# - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# - File -> Editor Type -> Script

import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(df.head())
print(df.info())

# city와 f4를 기준으로한 f5의 평균값
group_f5 = df.groupby(['city', 'f4']).mean().f5
print(group_f5)

# f5를 기준으로 상위 7개 값
round(sum(sorted(group_f5, reverse=True)[0:7]),2)
#결과값 : 643.68



# kaggle 답안
import pandas as pd

df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
df.head()

# city와 f4별 f5의 평균 값 (멀티인덱스 출력)
df = df.groupby(['city', 'f4'])[['f5']].mean()
print(df)

# dataframe 전환 후 상위 7개 출력
df = df.reset_index().sort_values('f5', ascending=False).head(7)
print(df)

# f7의 합계 (소수점 둘째자리까지)
round(df['f5'].sum(), 2)

#결과값 : 643.68