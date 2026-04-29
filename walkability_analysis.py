#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北京宋家庄地铁站步行可达性分析
课程作业 - GIS空间可视化

作者: [您的姓名]
日期: 2026年4月28日
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon
import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("开始北京宋家庄地铁站步行可达性分析...")

# ==============================================================================
# 1. 数据获取与网络构建
# ==============================================================================

print("1. 下载OpenStreetMap数据...")

# 设置研究区域（宋家庄地铁站为中心，半径1500米）
center_point = (39.8566, 116.4350)  # 宋家庄地铁站坐标
radius = 1500  # 1500米半径

# 下载路网数据
G = ox.graph_from_point(center_point, dist=radius, network_type='walk')

# 使用已知坐标（宋家庄地铁站）
# 如果地理编码失败，使用预设坐标
subway_coords = (39.8566, 116.4350)  # 宋家庄地铁站经纬度

print(f"地铁站位置: {subway_coords}")
print(f"网络节点数: {len(G.nodes())}")
print(f"网络边数: {len(G.edges())}")

# ==============================================================================
# 2. 网络分析与最短路径计算
# ==============================================================================

print("2. 进行网络分析...")

# 找到最近的网络节点 (注意: 参数顺序是 经度, 纬度)
orig_node = ox.distance.nearest_nodes(G, subway_coords[1], subway_coords[0])

# 设置步行速度 (1.2 m/s)
walking_speed = 1.2  # m/s

# 计算到所有节点的最短距离
shortest_distances = nx.shortest_path_length(G, source=orig_node, weight='length')

# 转换为GeoDataFrame便于分析
nodes_gdf = ox.graph_to_gdfs(G, edges=False)

# 添加距离信息
nodes_gdf['distance_to_station'] = nodes_gdf.index.map(shortest_distances).fillna(np.inf)

# 计算步行时间（分钟）
nodes_gdf['walking_time'] = (nodes_gdf['distance_to_station'] / walking_speed) / 60

print(f"最远可达距离: {nodes_gdf['distance_to_station'].max():.0f} 米")
print(f"最长步行时间: {nodes_gdf['walking_time'].max():.1f} 分钟")

# ==============================================================================
# 3. 生成等时圈
# ==============================================================================

print("3. 生成步行可达性等时圈...")

# 定义等时圈时间（分钟）
isochrone_times = [5, 10, 15, 20]

# 创建等时圈多边形
isochrones = []
for time in isochrone_times:
    # 获取在指定时间内可达的节点
    reachable_nodes = nodes_gdf[nodes_gdf['walking_time'] <= time]

    if len(reachable_nodes) > 2:
        # 创建凸包
        from shapely.ops import unary_union
        hull = unary_union(reachable_nodes.geometry).convex_hull
        isochrones.append({
            'time': time,
            'geometry': hull,
            'area': 0  # 面积将在投影转换后计算
        })

# 转换为GeoDataFrame
isochrones_gdf = gpd.GeoDataFrame(isochrones, crs='EPSG:4326')

# 保存原始地理坐标系版本用于可视化
isochrones_gdf_wgs84 = isochrones_gdf.copy()

# 转换为投影坐标系以计算准确面积 (UTM Zone 50N，适用于北京)
isochrones_gdf = isochrones_gdf.to_crs('EPSG:32650')

# 计算各等时圈的实际面积（平方米）
isochrones_gdf['area'] = isochrones_gdf.geometry.area

# ==============================================================================
# 4. 识别服务盲区
# ==============================================================================

print("4. 识别服务盲区...")

# 创建研究区域边界
study_area = Point(subway_coords[1], subway_coords[0]).buffer(radius / 111320)  # 转换为度

# 15分钟等时圈外的区域为服务盲区 (使用WGS84版本)
service_area_15min = isochrones_gdf_wgs84[isochrones_gdf_wgs84['time'] == 15]['geometry'].iloc[0]
blind_zones = study_area.difference(service_area_15min)

# ==============================================================================
# 5. 可视化结果
# ==============================================================================

print("5. 生成可视化图表...")

# 图表1: 研究区与路网分布图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 绘制路网
ox.plot_graph(G, ax=ax, node_size=0, edge_color='gray', edge_linewidth=0.5, show=False)

# 绘制地铁站位置
ax.plot(subway_coords[1], subway_coords[0], 'ro', markersize=10, label='宋家庄地铁站', zorder=5)

# 绘制研究区域边界
study_gdf = gpd.GeoDataFrame([{'geometry': study_area}], crs='EPSG:4326')
study_gdf.boundary.plot(ax=ax, color='blue', linestyle='--', alpha=0.7, label='研究区域边界')

# 设置图形属性
ax.set_title('北京宋家庄地铁站研究区域与路网分布', fontsize=16, fontweight='bold')
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加比例尺（简化版本）
ax.text(0.02, 0.02, '比例尺: 1:50000', transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'network_map.png'), dpi=300, bbox_inches='tight')
plt.close()

print("已保存: network_map.png")

# 图表2: 步行可达性等时圈图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 绘制路网背景
ox.plot_graph(G, ax=ax, node_size=0, edge_color='lightgray', edge_linewidth=0.3, show=False)

# 绘制等时圈 (使用WGS84坐标系用于可视化)
colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
labels = ['5分钟', '10分钟', '15分钟', '20分钟']

for i, (_, row) in enumerate(isochrones_gdf_wgs84.iterrows()):
    if hasattr(row.geometry, 'exterior'):
        x, y = row.geometry.exterior.xy
        ax.fill(x, y, alpha=0.3, color=colors[i], label=f'{row.time}分钟等时圈')
        ax.plot(x, y, color=colors[i], linewidth=2)

# 绘制地铁站
ax.plot(subway_coords[1], subway_coords[0], 'ro', markersize=12, label='宋家庄地铁站', zorder=5)

# 设置图形属性
ax.set_title('北京宋家庄地铁站步行可达性等时圈分析', fontsize=16, fontweight='bold')
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'isochrone_map.png'), dpi=300, bbox_inches='tight')
plt.close()

print("已保存: isochrone_map.png")

# 图表3: 服务盲区分析图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 绘制路网背景
ox.plot_graph(G, ax=ax, node_size=0, edge_color='lightgray', edge_linewidth=0.3, show=False)

# 绘制15分钟等时圈 (使用WGS84坐标系)
service_area = isochrones_gdf_wgs84[isochrones_gdf_wgs84['time'] == 15]
if len(service_area) > 0:
    service_geom = service_area.geometry.iloc[0]
    if hasattr(service_geom, 'exterior'):
        x, y = service_geom.exterior.xy
        ax.fill(x, y, alpha=0.3, color='green', label='15分钟服务范围内')
        ax.plot(x, y, color='green', linewidth=2)

# 绘制服务盲区
if not blind_zones.is_empty:
    if blind_zones.geom_type == 'Polygon':
        x, y = blind_zones.exterior.xy
        ax.fill(x, y, alpha=0.5, color='red', label='服务盲区')
    elif blind_zones.geom_type == 'MultiPolygon':
        for geom in blind_zones.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.5, color='red', label='服务盲区')

# 绘制地铁站
ax.plot(subway_coords[1], subway_coords[0], 'ro', markersize=12, label='宋家庄地铁站', zorder=5)

# 设置图形属性
ax.set_title('北京宋家庄地铁站服务盲区分析', fontsize=16, fontweight='bold')
ax.set_xlabel('经度')
ax.set_ylabel('纬度')

# 处理图例重复问题
handles, labels_legend = ax.get_legend_handles_labels()
by_label = dict(zip(labels_legend, handles))
ax.legend(by_label.values(), by_label.keys())

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'blind_zone_map.png'), dpi=300, bbox_inches='tight')
plt.close()

print("已保存: blind_zone_map.png")

# ==============================================================================
# 6. 统计分析与结果输出
# ==============================================================================

print("6. 生成统计分析...")

# 计算各等时圈覆盖面积
for _, row in isochrones_gdf.iterrows():
    print(f"{row.time}分钟等时圈覆盖面积: {row.area:.2f} 平方米")

# 计算服务盲区面积
total_area = study_area.area
service_area = isochrones_gdf[isochrones_gdf['time'] == 15]['area'].iloc[0] if len(isochrones_gdf[isochrones_gdf['time'] == 15]) > 0 else 0
blind_area = total_area - service_area

print(f"\n分析完成！所有文件已保存到: {output_dir}")

