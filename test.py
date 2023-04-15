import pandas as pd

df = pd.read_excel('data/Brozo.xlsx')[['Author username', 'Content']]
# 'Content' 컬럼에서 공백 문자열을 NaN 값으로 대체합니다.
df['Content'] = df['Content'].replace('', pd.np.nan)

# 'Content' 컬럼에서 NaN 값을 가지는 행을 제거합니다.
df.dropna(subset=['Content'], inplace=True)

# 'Content' 컬럼에서 문자열 앞뒤에 있는 공백을 제거합니다.
df['Content'] = df['Content'].str.strip()

all_text = ' '.join(df['Content'].tolist())

# 결합한 문자열을 공백으로 분리하여 단어 리스트를 만듭니다.
words = all_text.split()

# 단어 리스트의 길이를 세어서 단어 개수를 계산합니다.
num_words = len(words)

# 결과를 출력합니다.
print("Number of words in 'Content' column:", num_words)