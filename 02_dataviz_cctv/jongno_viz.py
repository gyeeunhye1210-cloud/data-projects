# 종로구 CCTV·보안등 시각화
# 데이터시각화이해와실습 2팀 최종보고서(2026-1)에 실린 코드를 옮겨 적은 파일입니다.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────
df_cctv = pd.read_csv('종로구_CCTV_현황_250226.csv', encoding='utf-8', header=1)
df_sec = pd.read_csv('서울시 종로구 동별 보안등 현황.csv', encoding='utf-8', header=1)
# ── 전처리 ────────────────────────────────────
# CCTV: 방범 목적만 필터링
df_cctv_bm = df_cctv[df_cctv['설치목적구분'] == '방범'].copy()

# 보안등: 이상 좌표 제거 (위도 0값)
df_sec = df_sec[(df_sec['위도'] > 37.0) & (df_sec['경도'] > 126.0)].copy()

print(f"   CCTV 전체: {len(df_cctv)}개소 / 방범 필터: {len(df_cctv_bm)}개소")
print(f"   보안등: {len(df_sec)}개")
# 출력: CCTV 전체: 1143개소 / 방범 필터: 646개소 / 보안등: 9476개


# ── 시각화 ① 설치목적별 CCTV 카메라 대수 막대그래프 ──
from matplotlib.patches import Patch

purpose_sum = (df_cctv
               .groupby('설치목적구분')['카메라대수']
               .sum()
               .sort_values(ascending=True))

colors = ['#2196F3' if p in ['방범', '국가유산방범'] else '#B0BEC5'
          for p in purpose_sum.index]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(purpose_sum.index, purpose_sum.values, color=colors, edgecolor='white')


for bar, val in zip(bars, purpose_sum.values):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f'{val:,}대', va='center', fontsize=11)

ax.set_xlabel('카메라 대수 (대)', fontsize=11)
ax.set_title('설치목적별 CCTV 카메라 대수', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, purpose_sum.max() * 1.18)
ax.spines[['top', 'right']].set_visible(False)

legend_elements = [Patch(facecolor='#2196F3', label='분석 대상 (방범)'),
                   Patch(facecolor='#B0BEC5', label='분석 제외')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('viz1_purpose_bar.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 시각화 ② 행정동별 보안등 현황 막대그래프 ──
sec_dong = df_sec.groupby('행정동').size().reset_index(name='보안등수')
sec_dong_sorted = sec_dong.sort_values('보안등수', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 6))
bar_colors = ['#F44336' if i == len(sec_dong_sorted)-1 else '#FFC107'
              for i in range(len(sec_dong_sorted))]
bars = ax.bar(range(len(sec_dong_sorted)), sec_dong_sorted['보안등수'],
              color=bar_colors, edgecolor='white')

for bar, val in zip(bars, sec_dong_sorted['보안등수']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:,}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(len(sec_dong_sorted)))
ax.set_xticklabels(sec_dong_sorted['행정동'], rotation=35, ha='right', fontsize=10)
ax.set_ylabel('보안등 수 (개)', fontsize=11)
ax.set_title('행정동별 보안등 분포 현황', fontsize=14, fontweight='bold', pad=15)
ax.spines[['top', 'right']].set_visible(False)

# 최하위 동 어노테이션
min_row = sec_dong_sorted.iloc[-1]
ax.annotate(f'⚠ 최하위\n{min_row["행정동"]} {min_row["보안등수"]:,}개',
            xy=(len(sec_dong_sorted)-1, min_row['보안등수']),
            xytext=(len(sec_dong_sorted)-3, min_row['보안등수'] + 200),
            fontsize=10, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

plt.tight_layout()
plt.savefig('viz2_dong_bar.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 시각화 ③ CCTV/보안등 설치 위치 산점도 지도 ──
m3 = folium.Map(location=[37.591, 126.990], zoom_start=13, tiles='CartoDB positron')

# 보안등 레이어
sec_layer = folium.FeatureGroup(name='🟡 보안등 (9,476개)', show=True)
for _, row in df_sec.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=2,
        color='#FFC107', fill=True, fill_opacity=0.5,
        popup=folium.Popup(
            f"<b>행정동:</b> {row['행정동']}<br>"
            f"<b>램프종류:</b> {row['램프종류']}<br>"
            f"<b>등주형태:</b> {row['등주형태']}",
            max_width=200)
    ).add_to(sec_layer)
sec_layer.add_to(m3)

# CCTV 방범 레이어
cctv_layer = folium.FeatureGroup(name='🔵 CCTV 방범 (646개소)', show=True)
for _, row in df_cctv_bm.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
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

# 주피터에서 바로 보기
from IPython.display import display
display(m3)


# ── 시각화 ④ 행정동별 단계구분도 ──
import requests

GEOJSON_URL = (
    "https://raw.githubusercontent.com/vuski/admdongkor/"
    "master/ver20230701/HangJeongDong_ver20230701.geojson"
)

geo_data = requests.get(GEOJSON_URL, timeout=15).json()

# 종로구만 필터링 (sgg 코드로)
geo_jongno = {
    "type": "FeatureCollection",
    "features": [
        f for f in geo_data['features']
        if f['properties'].get('sgg') == '11110'
    ]
}

# adm_nm에서 "서울특별시 종로구 " 제거하고 행정동명만 추출
for f in geo_jongno['features']:
    f['properties']['행정동'] = f['properties']['adm_nm'].replace('서울특별시 종로구 ', '')

print("종로구 행정동 수:", len(geo_jongno['features']))
print("행정동 목록:", [f['properties']['행정동'] for f in geo_jongno['features']])

# 보안등 행정동별 집계
sec_dong_map = df_sec.groupby('행정동').size().reset_index(name='보안등수')

m4 = folium.Map(location=[37.591, 126.990], zoom_start=13,
                tiles='CartoDB positron')

folium.Choropleth(
    geo_data=geo_jongno,
    data=sec_dong_map,
    columns=['행정동', '보안등수'],
    key_on='feature.properties.행정동',
    fill_color='RdYlGn',
    fill_opacity=0.75,
    line_opacity=0.5,
    legend_name='보안등 수 (개)',
    nan_fill_color='#eeeeee',
).add_to(m4)

folium.GeoJson(
    geo_jongno,
    style_function=lambda x: {'fillOpacity': 0, 'color': '#555', 'weight': 1},
    tooltip=folium.GeoJsonTooltip(
        fields=['행정동', ],
        aliases=['행정동:'],
    )
).add_to(m4)

m4.save('viz4_choropleth.html')
from IPython.display import display
display(m4)


# ── 시각화 ⑤ 인프라 편중 상태 분류 (산점도) ──
# 행정동별 가로등수·CCTV수와 인프라_상태를 담은 summary 표, 평균 avg_l·avg_c를 만드는 셀은
# 보고서에 캡처가 실리지 않아 여기에는 없습니다. 아래는 보고서에 실린 그리기 코드입니다.
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import platform

# 운영체제별 폰트 설정
if platform.system() == 'Darwin': # 맥(Mac)
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # 윈도우
    plt.rc('font', family='Malgun Gothic')
else: # 리눅스 (코랩 등)
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
# 결론 시각화: 가로등 vs CCTV 산점도
fig, ax = plt.subplots(figsize=(11, 8))

colors_map = {
    '✅ 양호':        '#2b7bba',
    '🔴 둘 다 부족':  '#cc2200',
    '🟡 가로등 부족': '#e6b800',
    '🟠 CCTV 부족':  '#e07000',
}

for 상태, group in summary.groupby('인프라_상태'):
    ax.scatter(group['가로등수'], group['CCTV수'],
               s=150, alpha=0.85,
               c=colors_map[상태], label=상태, zorder=3)
    for _, row in group.iterrows():
        ax.annotate(row['행정동'],
                    (row['가로등수'], row['CCTV수']),
                    fontsize=9, xytext=(5, 3),
                    textcoords='offset points')

# 평균선
ax.axvline(avg_l, color='gray', linestyle='--', linewidth=1, alpha=0.6, label=f'가로등 평균({avg_l:.0f}개)')
ax.axhline(avg_c, color='gray', linestyle=':',  linewidth=1, alpha=0.6, label=f'CCTV 평균({avg_c:.0f}대)')

ax.set_xlabel('가로등(보안등) 수 (개)', fontsize=12)
ax.set_ylabel('CCTV 수 (대)', fontsize=12)
ax.set_title('종로구 행정동별 가로등 vs CCTV 분포\n(빨강=둘 다 부족, 파랑=양호)', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('그래프_종합비교.png', dpi=150, bbox_inches='tight')
plt.show()
print('저장 완료: 그래프_종합비교.png')


# ── 시각화 ⑥ CCTV 및 가로등 위치 데이터 결합 히트맵 ──
m5 = folium.Map(location=[37.591, 126.990], zoom_start=13, tiles='CartoDB positron')

# 보안등 히트맵 데이터
sec_heat_list = df_sec[['위도', '경도']].assign(weight=1.0).values.tolist()

# CCTV 방범 히트맵 데이터 (카메라 대수 × 3 가중치)
cctv_heat_list = (df_cctv_bm[['위도', '경도', '카메라대수']]
                  .assign(카메라대수=lambda x: x['카메라대수'] * 3)
                  .values.tolist())

combined_heat = sec_heat_list + cctv_heat_list

gradient={0.4: '#FF0000', 0.7: '#FFFF00', 1.0: '#00C853'}

HeatMap(sec_heat_list,  name='🟡 보안등 밀도',
        radius=18, blur=15, gradient=gradient, show=False).add_to(m5)

HeatMap(cctv_heat_list, name='🔵 CCTV 밀도',
        radius=22, blur=18, gradient=gradient, show=False).add_to(m5)

HeatMap(combined_heat,  name='🟢 통합 안전 인프라 밀도 (기본 표시)',
        radius=20, blur=15, gradient=gradient, show=True).add_to(m5)

folium.LayerControl(collapsed=False).add_to(m5)

legend_html = """
<div style="position:fixed; bottom:40px; left:40px; z-index:1000;
     background:white; padding:12px 16px; border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:sans-serif;">
  <b style="font-size:13px;">안전 인프라 밀도</b><br>
  <div style="margin-top:8px;">
    <div style="width:130px; height:14px;
         background:linear-gradient(to right,#FF0000,#FFFF00,#00C853);
         border-radius:3px;"></div>
    <div style="display:flex; justify-content:space-between;
         width:130px; font-size:11px; margin-top:3px;">
      <span>취약</span><span>보통</span><span>안전</span>
    </div>
  </div>
</div>
"""
m5.get_root().html.add_child(folium.Element(legend_html))

m5.save('viz5_heatmap.html')
from IPython.display import display
display(m5)
