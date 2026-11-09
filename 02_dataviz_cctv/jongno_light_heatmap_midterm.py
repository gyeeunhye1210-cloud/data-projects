# 종로구 보안등 히트맵 (중간보고서 단계)
# 데이터시각화이해와실습 2팀 최종보고서(2026-1)에 실린 코드를 옮겨 적은 파일입니다.

# [1]
import pandas as pd
import folium
from folium.plugins import HeatMap
import os

print('✅ 라이브러리 로드 완료')

# [2]
# ================================================
# CSV 파일 경로 설정
# 이 노트북 파일과 같은 폴더에 CSV를 넣어두세요
# ================================================
LIGHT_CSV = '서울시 종로구 동별 보안등 현황.csv'

# 인코딩 자동 감지 (맥북은 cp949 오류 날 수 있어서 순서대로 시도)
for enc in ['utf-8-sig', 'euc-kr', 'cp949']:
    try:
        light = pd.read_csv(LIGHT_CSV, encoding=enc)
        print(f'✅ {enc} 인코딩으로 읽기 성공')
        break
    except UnicodeDecodeError:
        print(f'❌ {enc} 실패, 다음 인코딩 시도...')

print(f'\n📊 불러온 데이터 행 수: {len(light)}')
print(f'📋 컬럼 목록: {list(light.columns)}')
# 출력: euc-kr 인코딩으로 읽기 성공 / 불러온 데이터 행 수: 9480

# [3] 데이터 미리보기
light.head()

# [4] 위도, 경도 숫자형 변환
light['위도'] = pd.to_numeric(light['위도'], errors='coerce')
light['경도'] = pd.to_numeric(light['경도'], errors='coerce')

# 결측치 제거
before = len(light)
light = light.dropna(subset=['위도', '경도'])
after = len(light)

print(f'🧹 결측치 제거: {before - after}행 제거 → 남은 데이터: {after}행')

# [5] 중심 좌표 계산
center_lat = light['위도'].mean()
center_lon = light['경도'].mean()
print(f'📍 지도 중심 좌표: {center_lat:.4f}, {center_lon:.4f}')

# 지도 생성
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# 히트맵 추가
heat_data = light[['위도', '경도']].values.tolist()

HeatMap(
    heat_data,
    radius=18,
    blur=15,
    max_zoom=14,
    min_opacity=0.4
).add_to(m)

# HTML 저장
m.save('jongno_security_light_heatmap.html')
print('✅ jongno_security_light_heatmap.html 저장 완료')

# 셀 안에서 지도 바로 렌더링
m
