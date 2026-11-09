# 성북구 CCTV·보안등(안심귀갓길) 시각화
# 데이터시각화이해와실습 2팀 최종보고서(2026-1)에 실린 코드를 옮겨 적은 파일입니다.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium
from folium.plugins import HeatMap
import re, warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──────────────────────────────
# Mac    → 'AppleGothic'
# Windows → 'Malgun Gothic'
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────
df_cctv  = pd.read_csv('CCTV정보_서울성북구.csv', encoding='cp949')
df_ansim = pd.read_csv('서울시 안심귀갓길 안전시설물.csv', encoding='cp949')

# ── 전처리 ────────────────────────────────────
# 1) CCTV: 이상 좌표 제거 + 생활방범 필터
df_cctv_valid = df_cctv[
    (df_cctv['WGS84위도'] > 37.0) &
    (df_cctv['WGS84경도'] < 130)
].copy()
df_cctv_bm = df_cctv_valid[df_cctv_valid['설치목적구분'] == '생활방범'].copy()

# 2) 안심귀갓길: 성북구만 필터
df_sb = df_ansim[df_ansim['시군구명'].str.contains('성북', na=False)].copy()

# 3) WKT → 위도/경도 컬럼 분리
def extract_coords(wkt):
    m = re.search(r'POINT\s*\(\s*([0-9.]+)\s+([0-9.]+)\s*\)', str(wkt))
    if m:
        return float(m.group(2)), float(m.group(1))   # lat, lon
    return None, None

df_sb[['위도','경도']] = df_sb['포인트 wkt'].apply(
    lambda x: pd.Series(extract_coords(x))
)

# 4) 보안등(시설코드 305)만 추출
df_light = df_sb[df_sb['시설코드'] == 305].copy()


print(f"   CCTV 전체: {len(df_cctv_valid)}개소 / 생활방범 필터: {len(df_cctv_bm)}개소")
print(f"   안심귀갓길 성북구 전체: {len(df_sb)}개 / 보안등: {len(df_light)}개")


# ── 시각화 ① 설치목적별 CCTV 카메라 대수 막대그래프 ──
from matplotlib.patches import Patch

purpose_sum = (df_cctv_valid
               .groupby('설치목적구분')['카메라대수']
               .sum()
               .sort_values(ascending=True))

colors = ['#2196F3' if p == '생활방범' else '#B0BEC5'
          for p in purpose_sum.index]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(purpose_sum.index, purpose_sum.values,
               color=colors, edgecolor='white')

for bar, val in zip(bars, purpose_sum.values):
    ax.text(bar.get_width() + 15,
            bar.get_y() + bar.get_height() / 2,
            f'{val:,}대', va='center', fontsize=11)

ax.set_xlabel('카메라 대수 (대)', fontsize=11)
ax.set_title('설치목적별 CCTV 카메라 대수 (성북구)', fontsize=14,
             fontweight='bold', pad=15)
ax.set_xlim(0, purpose_sum.max() * 1.18)
ax.spines[['top','right']].set_visible(False)

legend_elements = [
    Patch(facecolor='#2196F3', label='분석 대상 (생활방범)'),
    Patch(facecolor='#B0BEC5', label='분석 제외')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('viz1_purpose_bar.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 시각화 ② 읍면동별 보안등 현황 막대그래프 ──
light_dong = (df_light.groupby('읍면동명')
              .size()
              .reset_index(name='보안등수')
              .sort_values('보안등수', ascending=False)
              .reset_index(drop=True))

fig, ax = plt.subplots(figsize=(13, 6))

bar_colors = []
min_val = light_dong['보안등수'].min()
for v in light_dong['보안등수']:
    if v == min_val:
        bar_colors.append('#F44336')
    elif v == light_dong['보안등수'].max():
        bar_colors.append('#4CAF50')
    else:
        bar_colors.append('#FFC107')

bars = ax.bar(range(len(light_dong)), light_dong['보안등수'],
              color=bar_colors, edgecolor='white')

for bar, val in zip(bars, light_dong['보안등수']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:,}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(len(light_dong)))
ax.set_xticklabels(light_dong['읍면동명'], rotation=35, ha='right', fontsize=10)
ax.set_ylabel('보안등 수 (개)', fontsize=11)
ax.set_title('읍면동별 보안등 분포 현황 (성북구 안심귀갓길)', fontsize=14,
             fontweight='bold', pad=15)
ax.spines[['top','right']].set_visible(False)

# 최하위 어노테이션
min_row = light_dong[light_dong['보안등수'] == min_val].iloc[0]
min_idx = light_dong.index[light_dong['읍면동명'] == min_row['읍면동명']].tolist()[0]
ax.annotate(f'⚠ 최하위\n{min_row["읍면동명"]} {min_row["보안등수"]}개',
            xy=(min_idx, min_row['보안등수']),
            xytext=(min_idx - 2, min_row['보안등수'] + 10),
            fontsize=10, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

plt.tight_layout()
plt.savefig('viz2_dong_bar.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 시각화 ③ CCTV/보안등 설치 위치 산점도 지도 ──
MAP_CENTER = [37.6025, 127.0290]

m3 = folium.Map(location=MAP_CENTER, zoom_start=13,
                tiles='CartoDB positron')

# 보안등 레이어
sec_layer = folium.FeatureGroup(name='🟡 보안등 (330개)', show=True)
for _, row in df_light.iterrows():
    if pd.notna(row['위도']) and pd.notna(row['경도']):
        folium.CircleMarker(
            location=[row['위도'], row['경도']],
            radius=4,
            color='#FFC107', fill=True, fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>읍면동:</b> {row['읍면동명']}<br>"
                f"<b>안심귀갓길:</b> {row['안심귀갓길 명']}<br>"
                f"<b>관리기관:</b> {row['관리기관']}",
                max_width=220)
        ).add_to(sec_layer)
sec_layer.add_to(m3)

# 생활방범 CCTV 레이어
cctv_layer = folium.FeatureGroup(name='🔵 CCTV 생활방범 (1,331개소)', show=True)
for _, row in df_cctv_bm.iterrows():
    folium.CircleMarker(
        location=[row['WGS84위도'], row['WGS84경도']],
        radius=max(3, row['카메라대수'] * 1.2),
        color='#1565C0', fill=True, fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>주소:</b> {row['소재지도로명주소']}<br>"
            f"<b>카메라 대수:</b> {row['카메라대수']}대<br>"
            f"<b>설치목적:</b> {row['설치목적구분']}",
            max_width=220)
    ).add_to(cctv_layer)
cctv_layer.add_to(m3)

folium.LayerControl(collapsed=False).add_to(m3)
m3.save('viz3_scatter_map.html')

from IPython.display import display
display(m3)


# ── 시각화 ④ 행정동별 단계구분도 ──
import requests

GEOJSON_URL = (
    "https://raw.githubusercontent.com/vuski/admdongkor/"
    "master/ver20230701/HangJeongDong_ver20230701.geojson"
)

try:
    geo_data = requests.get(GEOJSON_URL, timeout=15).json()

    # 성북구만 필터링 (sgg 코드: 11290)
    geo_sb = {
        "type": "FeatureCollection",
        "features": [
            f for f in geo_data['features']
            if f['properties'].get('sgg') == '11290'
        ]
    }

    print(f"성북구 행정동 수: {len(geo_sb['features'])}")

    for f in geo_sb['features']:
        full = f['properties']['adm_nm']
        f['properties']['읍면동명'] = full.replace('서울특별시 성북구 ', '')

    print("행정동 목록:", [f['properties']['읍면동명'] for f in geo_sb['features']])

    light_dong_map = (df_light.groupby('읍면동명')
                      .size()
                      .reset_index(name='보안등수'))

    m4 = folium.Map(location=[37.6025, 127.0290], zoom_start=13,
                    tiles='CartoDB positron')

    folium.Choropleth(
        geo_data=geo_sb,
        data=light_dong_map,
        columns=['읍면동명', '보안등수'],
        key_on='feature.properties.읍면동명',
        fill_color='RdYlGn',
        fill_opacity=0.75,
        line_opacity=0.5,
        legend_name='보안등 수 (개)',
        nan_fill_color='#eeeeee',
    ).add_to(m4)

    folium.GeoJson(
        geo_sb,
        style_function=lambda x: {
            'fillOpacity': 0, 'color': '#555', 'weight': 1
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['읍면동명'],
            aliases=['읍면동:']
        )
    ).add_to(m4)

    m4.save('viz4_choropleth.html')
    from IPython.display import display
    display(m4)
    print("✅ viz4_choropleth.html 저장 완료")
# 보고서 캡처에는 try에 대응하는 except 부분이 실려 있지 않아, 실행되도록 최소한으로만 덧붙였습니다.
except Exception as e:
    print("GeoJSON을 불러오지 못했습니다:", e)


# ── 시각화 ⑤ 인프라 편중 상태 분류 (4분면 산점도) ──
# 행정동별 집계표 cctv_by_dong(CCTV_대수), light_by_dong(보안등_수)을 만드는 셀은
# 보고서에 캡처가 실리지 않아 여기에는 없습니다. 아래는 보고서에 실린 그리기 코드입니다.
merged = pd.merge(cctv_by_dong, light_by_dong, on='행정동', how='outer').fillna(0)
cctv_mean = merged['CCTV_대수'].mean()
light_mean = merged['보안등_수'].mean()

fig, ax = plt.subplots(figsize=(12, 9))

# 사분면 배경
ax.axhspan(cctv_mean, merged['CCTV_대수'].max() * 1.12, xmin=0.5, alpha=0.07, color='green')
ax.axhspan(0, cctv_mean, xmax=0.5, alpha=0.07, color='red')
ax.axhspan(0, cctv_mean, xmin=0.5, alpha=0.04, color='orange')
ax.axhspan(cctv_mean, merged['CCTV_대수'].max() * 1.12, xmax=0.5, alpha=0.04, color='blue')

# 평균선
ax.axhline(cctv_mean, color='gray', linestyle='--', alpha=0.6, linewidth=1.2,
           label=f'CCTV 평균 {cctv_mean:.0f}대')
ax.axvline(light_mean, color='gray', linestyle=':', alpha=0.6, linewidth=1.2,
           label=f'보안등 평균 {light_mean:.0f}개')

# 사분면 라벨
ymax = merged['CCTV_대수'].max()
ax.text(light_mean * 0.05, ymax * 1.07, '② CCTV 충분\n보안등 부족', fontsize=9, color='navy', alpha=0.8)
ax.text(light_mean * 1.05, ymax * 1.07, '① 균형 지역', fontsize=9, color='darkgreen', alpha=0.8)
ax.text(light_mean * 0.05, cctv_mean * 0.05, '③ 이중 취약', fontsize=9, color='darkred', alpha=0.8)
ax.text(light_mean * 1.05, cctv_mean * 0.05, '④ 보안등 충분\nCCTV 부족', fontsize=9, color='darkorange', alpha=0.8)

# 점 색상
def get_color(row):
    if row['CCTV_대수'] >= cctv_mean and row['보안등_수'] >= light_mean:
        return '#27AE60'
    elif row['CCTV_대수'] >= cctv_mean and row['보안등_수'] < light_mean:
        return '#2980B9'
    elif row['CCTV_대수'] < cctv_mean and row['보안등_수'] >= light_mean:
        return '#E67E22'
    else:
        return '#E74C3C'

colors_s = merged.apply(get_color, axis=1)
ax.scatter(merged['보안등_수'], merged['CCTV_대수'],
           c=colors_s, s=130, edgecolors='white', linewidth=0.8, zorder=5)

for _, row in merged.iterrows():
    ax.annotate(row['행정동'], (row['보안등_수'], row['CCTV_대수']),
                textcoords='offset points', xytext=(7, 4), fontsize=8.5, color='#2C3E50')

ax.set_xlabel(f'보안등 수 (개)', fontsize=12)
ax.set_ylabel(f'CCTV 대수 (대)', fontsize=12)
ax.set_title('성북구 행정동별 가로등 vs CCTV 분포 (4분면 분류)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(-3, merged['보안등_수'].max() * 1.25)
ax.set_ylim(-10, merged['CCTV_대수'].max() * 1.18)

legend_el = [
    Patch(facecolor='#27AE60', label='① 균형 지역 (둘 다 충분)'),
    Patch(facecolor='#2980B9', label='② CCTV 충분, 보안등 부족'),
    Patch(facecolor='#E67E22', label='④ 보안등 충분, CCTV 부족'),
    Patch(facecolor='#E74C3C', label='③ 이중 취약 (둘 다 부족)'),
]
ax.legend(handles=legend_el, loc='upper left', fontsize=9, framealpha=0.9)
ax.legend(handles=legend_el + [
    plt.Line2D([0],[0], color='gray', linestyle='--', label=f'CCTV 평균 {cctv_mean:.0f}대'),
    plt.Line2D([0],[0], color='gray', linestyle=':', label=f'보안등 평균 {light_mean:.0f}개'),
], loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('시각화4_인프라편중산점도_성북구.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 저장 완료: 시각화4_인프라편중산점도_성북구.png")
