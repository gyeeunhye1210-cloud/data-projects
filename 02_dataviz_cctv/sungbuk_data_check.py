# 성북구 데이터 확인 (분석 초기 단계)
# 데이터시각화이해와실습 2팀 최종보고서(2026-1)에 실린 코드를 옮겨 적은 파일입니다.
import re
import pandas as pd

# [8] ── CCTV 데이터 로드 ──────────────────────────
cctv_raw = pd.read_csv('CCTV정보_서울성북구.csv', encoding='cp949')

# 성북구 좌표 범위로 이상치 제거
cctv = cctv_raw[
    (cctv_raw['WGS84위도'] >= 37.57) & (cctv_raw['WGS84위도'] <= 37.65) &
    (cctv_raw['WGS84경도'] >= 126.95) & (cctv_raw['WGS84경도'] <= 127.10)
].copy()

print(f"✅ CCTV 원본: {len(cctv_raw)}개 → 유효(이상치 제거): {len(cctv)}개소")
print(f"   총 카메라 대수: {cctv['카메라대수'].sum()}대")
print(f"   설치목적 종류: {list(cctv['설치목적구분'].unique())}")
cctv.head(3)
# 출력: CCTV 원본: 2366개 → 유효(이상치 제거): 2354개소 / 총 카메라 대수: 5542대

# [9] ── 보안등 데이터 로드 (안심귀갓길 시설코드 305) ──────────
ansim = pd.read_csv('서울시 안심귀갓길 안전시설물.csv', encoding='cp949')
light_raw = ansim[
    (ansim['시군구명'] == '서울특별시 성북구') &
    (ansim['시설코드'] == 305)
].copy()

# POINT(경도 위도) 문자열에서 좌표 추출
def parse_wkt(wkt):
    nums = re.findall(r'[\d.]+', str(wkt))
    if len(nums) >= 2:
        return float(nums[1]), float(nums[0])  # 위도, 경도
    return None, None

light_raw[['위도', '경도']] = light_raw['포인트 wkt'].apply(
    lambda x: pd.Series(parse_wkt(x))
)
light = light_raw.dropna(subset=['위도', '경도'])

print(f"✅ 보안등 유효 데이터: {len(light)}개")
print(f"   총 설치 대수: {int(light['설치대수'].sum())}대")
print("   ⚠️  안심귀갓길 구간 내 보안등만 포함된 데이터입니다.")
light[['읍면동명', '설치대수', '위도', '경도']].head(3)
# 출력: 보안등 유효 데이터: 330개 / 총 설치 대수: 374대
