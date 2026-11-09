# KOD-FedGNN 신규 사용자 음악 추천 파이프라인 (구현 예시)
# 2026-1 기계학습 Mini Research Project, Colab 노트북 「기계학습 연구.ipynb」의 코드를 옮긴 파일입니다.
# FedGNN 예측값은 실제 연합학습 결과가 아니라 시뮬레이션 값(Preference_score)으로 대체했습니다.
# 온보딩 응답(feedback)도 실제 사용자 응답이 아니라 무작위로 만든 예시입니다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

# 1. 데이터 불러오기
df = pd.read_csv("spotify_tracks.csv", encoding="latin1", low_memory=False)

# 2. 컬럼명 정리
df = df.rename(columns={
    "name": "track_name",
    "track": "track_name",
    "artist_name": "artists",
    "artist": "artists"
})

# 3. 사용할 오디오 피처
features = [
    "danceability",
    "energy",
    "tempo",
    "valence",
    "acousticness"
]

# 4. 필요한 컬럼 존재 확인
missing_cols = [col for col in features if col not in df.columns]

if missing_cols:
    raise ValueError(f"다음 컬럼이 데이터에 없습니다: {missing_cols}")

# 5. 오디오 피처를 숫자형으로 변환
# 숫자가 아닌 값(FALSE, TRUE, 문자열 등)은 NaN으로 처리
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. 결측값 제거
df = df.dropna(subset=features).reset_index(drop=True)

# 7. 이상값 제거
# Spotify 오디오 피처 대부분은 0~1 범위이고, tempo는 BPM 범위로 제한
df = df[
    (df["danceability"] >= 0) & (df["danceability"] <= 1) &
    (df["energy"] >= 0) & (df["energy"] <= 1) &
    (df["valence"] >= 0) & (df["valence"] <= 1) &
    (df["acousticness"] >= 0) & (df["acousticness"] <= 1) &
    (df["tempo"] >= 50) & (df["tempo"] <= 220)
].reset_index(drop=True)

# 8. 곡 제목/아티스트 결측값 처리
if "track_name" not in df.columns:
    df["track_name"] = [f"Song_{i}" for i in range(len(df))]
else:
    df["track_name"] = df["track_name"].fillna("Unknown Song")

if "artists" not in df.columns:
    df["artists"] = "Unknown Artist"
else:
    df["artists"] = df["artists"].fillna("Unknown Artist")

# 9. 정규화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])

# 10. K-means 군집화
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
# 군집 시각화
# 군집 시각화

plt.figure(figsize=(8,6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["cluster"]
)

plt.xlabel("Danceability")
plt.ylabel("Energy")

plt.title("K-Means Music Clustering")

plt.show()
# 11. 온보딩 곡 생성: 각 군집에서 2곡씩 선택
# 총 10곡 생성
onboarding_df = (
    df.groupby("cluster", group_keys=False)
      .apply(lambda x: x.sample(n=min(2, len(x)), random_state=42))
      .reset_index(drop=True)
)

print("=== 신규 사용자 온보딩 데이터 ===")
print(onboarding_df[["track_name", "cluster"]])

# 12. 온보딩 응답 시뮬레이션
# 실제 서비스에서는 사용자가 직접 1/0 응답
np.random.seed(42)
onboarding_df["feedback"] = np.random.choice([0, 1], size=len(onboarding_df))

print("=== 사용자 피드백 결과 ===")
print(onboarding_df[["track_name", "feedback"]])

# 13. 의사결정나무 학습
X_onboard = scaler.transform(onboarding_df[features])
y_onboard = onboarding_df["feedback"]

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

dt.fit(X_onboard, y_onboard)

# 14. 전체 곡에 대한 의사결정나무 선호 확률
# predict_proba 결과에서 class 1의 위치를 안전하게 찾기
proba = dt.predict_proba(X_scaled)

if 1 in dt.classes_:
    like_class_index = list(dt.classes_).index(1)
    df["P_DT"] = proba[:, like_class_index]
else:
    df["P_DT"] = 0

# 15. 신규 사용자 군집 설정
# 좋아요가 가장 많이 나온 군집을 사용자 초기 군집으로 선택
liked_clusters = onboarding_df[onboarding_df["feedback"] == 1]["cluster"]

if len(liked_clusters) > 0:
    user_cluster = liked_clusters.mode()[0]
else:
    user_cluster = onboarding_df["cluster"].mode()[0]

cluster_center = kmeans.cluster_centers_[user_cluster].reshape(1, -1)

# 16. 군집 유사도 Sim 계산
df["Sim"] = cosine_similarity(X_scaled, cluster_center).flatten()

# 17. Flow 점수 계산
# 첫 번째 추천 기준곡은 온보딩에서 좋아요를 받은 곡 중 하나로 설정
liked_onboarding = onboarding_df[onboarding_df["feedback"] == 1]

if len(liked_onboarding) > 0:
    prev_song_raw = liked_onboarding[features].iloc[0].values.reshape(1, -1)
else:
    prev_song_raw = onboarding_df[features].iloc[0].values.reshape(1, -1)

prev_song = scaler.transform(prev_song_raw)[0]

def flow_score(x, prev):
    tempo_diff = abs(x[2] - prev[2])
    energy_diff = abs(x[1] - prev[1])
    valence_diff = abs(x[3] - prev[3])
    return 1 / (1 + tempo_diff + energy_diff + valence_diff)

df["Flow"] = [flow_score(x, prev_song) for x in X_scaled]

# 18. FedGNN 예측값 예시
# 실제 FedGNN 학습값이 없으므로 시뮬레이션 값으로 대체
np.random.seed(42)
df["Preference_score"] = (     0.6 * df["P_DT"]   + 0.4 * df["Sim"])

# 19. 최종 추천 점수 계산
alpha = 0.4
beta = 0.3
gamma = 0.2
delta = 0.1

df["Final_Score"] = (
    alpha * df["Preference_score"]
    + beta * df["P_DT"]
    + gamma * df["Sim"]
    + delta * df["Flow"]
)

# 20. Top-10 추천 결과
top10 = df.sort_values("Final_Score", ascending=False).head(10)

top10_result = top10[
    ["track_name", "artists", "Final_Score", "P_DT", "Sim", "Flow", "cluster"]
].copy()

top10_result = top10_result.rename(columns={
    "track_name": "곡",
    "artists": "아티스트",
    "Final_Score": "최종 추천 점수",
    "P_DT": "의사결정나무 선호 확률",
    "Sim": "군집 유사도",
    "Flow": "곡 흐름 점수",
    "cluster": "Cluster"
})

print("Top-10 추천 결과")
print(top10_result)

#Precision@K 계산
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)

    hits = len([item for item in recommended_k if item in relevant_set])

    return hits / k

# F1-score 계산

y_true = y_onboard
y_pred = dt.predict(X_onboard)

f1 = f1_score(y_true, y_pred)

print("F1-score:", f1)

# 예시용 relevant item
relevant_items = top10["track_name"].head(3).tolist()

recommended_items = top10["track_name"].tolist()

precision_k = precision_at_k(
    recommended_items,
    relevant_items,
    k=5
)

print("Precision@5:", precision_k)
# 21. 온보딩 데이터 예시 표
onboarding_table = onboarding_df[
    ["track_name", "cluster", "danceability", "energy", "tempo", "valence", "acousticness", "feedback"]
].copy()

onboarding_table = onboarding_table.rename(columns={
    "track_name": "곡",
    "cluster": "Cluster",
    "danceability": "Danceability",
    "energy": "Energy",
    "tempo": "Tempo",
    "valence": "Valence",
    "acousticness": "Acousticness",
    "feedback": "응답"
})

print("\n온보딩 데이터 예시")
print(onboarding_table)
