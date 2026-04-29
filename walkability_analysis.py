#北京宋家庄地铁站步行可达性分析
import os
import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
from typing import Any, Tuple, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
WGS84_CRS = 'EPSG:4326'
UTM_CRS = 'EPSG:32650'
WALKING_SPEED = 1.2  # m/s


def build_network(center_point: tuple, radius: int, network_type: str = 'walk'):
    """
    构建步行网络
    
    Args:
        center_point: 中心点坐标 (纬度, 经度)
        radius: 搜索半径（米）
        network_type: 网络类型，默认'walk'（步行）
    
    Returns:
        G: OSMNx路网图
        subway_coords: 地铁站坐标 (纬度, 经度)
    """
    print("1. 下载OpenStreetMap数据...")
    
    subway_coords = center_point
    G = ox.graph_from_point(center_point, dist=radius, network_type=network_type)
    
    print(f"地铁站位置: {subway_coords}")
    print(f"网络节点数: {len(G.nodes())}")
    print(f"网络边数: {len(G.edges())}")
    
    return G, subway_coords


def compute_accessibility(G: nx.Graph, subway_coords: tuple, walking_speed: float = WALKING_SPEED):
    """
    计算最短路径与步行时间
    
    Args:
        G: OSMNx路网图
        subway_coords: 地铁站坐标 (纬度, 经度)
        walking_speed: 步行速度 (m/s)
    
    Returns:
        nodes_gdf: 包含距离和时间信息的节点GeoDataFrame
        orig_node: 起点节点ID
    """
    print("2. 进行网络分析...")
    
    # 找到最近的网络节点 (参数顺序: 经度, 纬度)
    orig_node = ox.distance.nearest_nodes(G, subway_coords[1], subway_coords[0])
    
    # 计算到所有节点的最短距离
    shortest_distances = nx.shortest_path_length(G, source=orig_node, weight='length')
    
    # 转换为GeoDataFrame
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    
    # 添加距离信息
    nodes_gdf['distance_to_station'] = nodes_gdf.index.map(shortest_distances).fillna(np.inf)
    
    # 计算步行时间（分钟）
    nodes_gdf['walking_time'] = (nodes_gdf['distance_to_station'] / walking_speed) / 60
    
    print(f"最远可达距离: {nodes_gdf['distance_to_station'].max():.0f} 米")
    print(f"最长步行时间: {nodes_gdf['walking_time'].max():.1f} 分钟")
    
    return nodes_gdf, orig_node


def generate_isochrones(nodes_gdf: gpd.GeoDataFrame, isochrone_times: list):
    """
    生成等时圈
    
    Args:
        nodes_gdf: 包含步行时间信息的节点GeoDataFrame
        isochrone_times: 等时圈时间列表（分钟）
    
    Returns:
        isochrones_gdf: WGS84坐标系的等时圈GeoDataFrame（用于可视化）
    """
    print("3. 生成步行可达性等时圈...")
    
    isochrones = []
    for time in isochrone_times:
        # 获取在指定时间内可达的节点
        reachable_nodes = nodes_gdf[nodes_gdf['walking_time'] <= time]
        
        if len(reachable_nodes) > 2:
            # 创建凸包
            hull = unary_union(reachable_nodes.geometry).convex_hull
            isochrones.append({
                'time': time,
                'geometry': hull,
                'area': 0  # 面积将在投影转换后计算
            })
    
    # 转换为GeoDataFrame (WGS84坐标系)
    isochrones_gdf = gpd.GeoDataFrame(isochrones, crs=WGS84_CRS)
    
    return isochrones_gdf


def calculate_area(isochrones_gdf: gpd.GeoDataFrame):
    """
    计算投影面积（UTM坐标系）
    
    Args:
        isochrones_gdf: WGS84坐标系的等时圈GeoDataFrame
    
    Returns:
        isochrones_gdf: 包含面积的等时圈GeoDataFrame (UTM坐标系)
    """
    # 转换为UTM坐标系计算面积
    isochrones_utm = isochrones_gdf.to_crs(UTM_CRS)
    isochrones_utm['area'] = isochrones_utm.geometry.area
    
    return isochrones_utm


def analyze_blind_zone(subway_coords: tuple, radius: int, isochrones_gdf: gpd.GeoDataFrame):
    """
    计算服务盲区

    Args:
        subway_coords: 地铁站坐标 (纬度, 经度)
        radius: 研究区域半径（米）
        isochrones_gdf: WGS84坐标系的等时圈GeoDataFrame

    Returns:
        study_area: 研究区域多边形 (WGS84坐标系)
        blind_zones: 服务盲区多边形 (WGS84坐标系)
        total_area_m2: 研究区域总面积（平方米）
    """
    # 创建研究区域边界 (WGS84坐标系)
    study_area = Point(subway_coords[1], subway_coords[0]).buffer(radius / 111320)
    study_area_gdf = gpd.GeoDataFrame([{'geometry': study_area}], crs=WGS84_CRS)

    # 转换研究区域和15分钟等时圈到UTM坐标系
    study_area_utm = study_area_gdf.to_crs(UTM_CRS)
    service_area_15min = isochrones_gdf[isochrones_gdf['time'] == 15]['geometry'].iloc[0]
    service_area_utm = gpd.GeoDataFrame([{'geometry': service_area_15min}], crs=WGS84_CRS).to_crs(UTM_CRS)

    # 计算服务盲区
    blind_zones_utm = study_area_utm.geometry.iloc[0].difference(service_area_utm.geometry.iloc[0])

    # 计算研究区域总面积
    total_area_m2 = study_area_utm.geometry.area.iloc[0]

    # 将服务盲区转换回WGS84坐标系
    blind_zones = gpd.GeoSeries([blind_zones_utm], crs=UTM_CRS).to_crs(WGS84_CRS).iloc[0]

    return study_area, blind_zones, total_area_m2


def plot_network_map(G: nx.Graph, subway_coords: tuple, study_area: Any, output_dir: str):
    """
    绘制研究区与路网分布图
    
    Args:
        G: OSMNx路网图
        subway_coords: 地铁站坐标
        study_area: 研究区域多边形
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制路网
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='gray', edge_linewidth=0.5, show=False)
    
    # 绘制地铁站位置
    ax.plot(subway_coords[1], subway_coords[0], 'ro', markersize=10, 
            label='宋家庄地铁站', zorder=5)
    
    # 绘制研究区域边界
    study_gdf = gpd.GeoDataFrame([{'geometry': study_area}], crs=WGS84_CRS)
    study_gdf.boundary.plot(ax=ax, color='blue', linestyle='--', alpha=0.7, 
                            label='研究区域边界')
    
    # 设置图形属性
    ax.set_title('北京宋家庄地铁站研究区域与路网分布', fontsize=16, fontweight='bold')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加比例尺
    ax.text(0.02, 0.02, '比例尺: 1:50000', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("已保存: network_map.png")


def plot_isochrone_map(G: nx.Graph, subway_coords: tuple, 
                       isochrones_gdf: gpd.GeoDataFrame, output_dir: str):
    """
    绘制步行可达性等时圈图
    
    Args:
        G: OSMNx路网图
        subway_coords: 地铁站坐标
        isochrones_gdf: WGS84坐标系的等时圈GeoDataFrame
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制路网背景
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='lightgray', edge_linewidth=0.3, show=False)
    
    # 绘制等时圈
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
    
    for i, (_, row) in enumerate(isochrones_gdf.iterrows()):
        if hasattr(row.geometry, 'exterior'):
            x, y = row.geometry.exterior.xy
            ax.fill(x, y, alpha=0.3, color=colors[i], label=f'{row.time}分钟等时圈')
            ax.plot(x, y, color=colors[i], linewidth=2)
    
    # 绘制地铁站
    ax.plot(subway_coords[1], subway_coords[0], 'ro', markersize=12, 
            label='宋家庄地铁站', zorder=5)
    
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

def plot_blind_zone_map(G: nx.Graph, subway_coords: tuple, 
                        isochrones_gdf: gpd.GeoDataFrame, 
                        blind_zones: Any, output_dir: str):
    """
    完全无重叠版服务盲区图（推荐最终提交用）
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 1. 路网背景
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='#e0e0e0',
                  edge_linewidth=0.5, show=False, close=False)

    # 2. 获取15分钟服务区
    service_area = isochrones_gdf[isochrones_gdf['time'] == 15]

    service_geom = None
    if len(service_area) > 0:
        service_geom = service_area.geometry.iloc[0]

    # 3. 👉 关键：从盲区中减去服务区（彻底消除重叠）
    clean_blind = blind_zones
    if service_geom is not None:
        try:
            clean_blind = blind_zones.difference(service_geom)
        except:
            pass  # 防止拓扑错误崩溃

    # 4. 画盲区（真正不重叠的）
    if not clean_blind.is_empty:
        if clean_blind.geom_type == 'Polygon':
            x, y = clean_blind.exterior.xy
            ax.fill(x, y, color='#ff4d4d', alpha=0.5,
                    label='服务盲区 (>15min)', zorder=1)
            ax.fill(x, y, color='none', edgecolor='#c0392b',
                    hatch='///', alpha=0.4, zorder=1)

        elif clean_blind.geom_type == 'MultiPolygon':
            for i, geom in enumerate(clean_blind.geoms):
                x, y = geom.exterior.xy
                label = '服务盲区 (>15min)' if i == 0 else ""
                ax.fill(x, y, color='#ff4d4d', alpha=0.5,
                        label=label, zorder=1)
                ax.fill(x, y, color='none', edgecolor='#c0392b',
                        hatch='///', alpha=0.4, zorder=1)

    # 5. 画服务区（完全独立）
    if service_geom is not None:
        if hasattr(service_geom, 'exterior'):
            x, y = service_geom.exterior.xy
            ax.fill(x, y, color='#2ecc71', alpha=0.8,
                    label='15分钟服务范围', zorder=2)
            ax.plot(x, y, color='#27ae60',
                    linewidth=2.5, zorder=3)

    # 6. 地铁站
    ax.scatter(subway_coords[1], subway_coords[0],
               color='yellow', edgecolor='black',
               s=250, marker='*',
               label='宋家庄地铁站', zorder=5)

    # 7. 图形优化
    ax.set_title('北京宋家庄地铁站服务盲区分析',
                 fontsize=18, fontweight='bold')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper right', fontsize=11)

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blind_zone_map_final.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print("已保存: blind_zone_map_final.png")


def plot_maps(G, subway_coords, isochrones_gdf, study_area, blind_zones, output_dir: str):
    """
    生成三张可视化地图
    
    Args:
        G: OSMNx路网图
        subway_coords: 地铁站坐标
        isochrones_gdf: WGS84坐标系的等时圈GeoDataFrame
        study_area: 研究区域多边形
        blind_zones: 服务盲区多边形
        output_dir: 输出目录
    """
    print("5. 生成可视化图表...")
    
    plot_network_map(G, subway_coords, study_area, output_dir)
    plot_isochrone_map(G, subway_coords, isochrones_gdf, output_dir)
    plot_blind_zone_map(G, subway_coords, isochrones_gdf, blind_zones, output_dir)


def print_statistics(isochrones_utm: gpd.GeoDataFrame, total_area_m2: float):
    """
    输出统计分析结果
    
    Args:
        isochrones_utm: UTM坐标系的等时圈GeoDataFrame
        total_area_m2: 研究区域总面积（平方米）
    """
    print("6. 生成统计分析...")
    
    # 各等时圈覆盖面积
    for _, row in isochrones_utm.iterrows():
        print(f"{row.time}分钟等时圈覆盖面积: {row.area:.2f} 平方米")
    
    # 服务盲区计算
    service_area_15min_m2 = isochrones_utm[isochrones_utm['time'] == 15]['area'].iloc[0] \
        if len(isochrones_utm[isochrones_utm['time'] == 15]) > 0 else 0
    blind_area_m2 = total_area_m2 - service_area_15min_m2
    
    print(f"研究区域总面积: {total_area_m2:.2f} 平方米")
    print(f"15分钟服务范围面积: {service_area_15min_m2:.2f} 平方米")
    print(f"服务盲区面积: {blind_area_m2:.2f} 平方米")
    print(f"服务覆盖率: {service_area_15min_m2/total_area_m2*100:.1f}%")


def main():
    """主函数"""
    # 参数设置
    center_point = (39.8566, 116.4350)  # 宋家庄地铁站坐标
    radius = 1500  # 研究半径（米）
    isochrone_times = [5, 10, 15, 20]  # 等时圈时间（分钟）
    
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("开始北京宋家庄地铁站步行可达性分析...")
    
    # 1. 构建路网
    G, subway_coords = build_network(center_point, radius)
    
    # 2. 计算可达性
    nodes_gdf, orig_node = compute_accessibility(G, subway_coords)
    
    # 3. 生成等时圈 (WGS84坐标系，用于可视化)
    isochrones_gdf = generate_isochrones(nodes_gdf, isochrone_times)
    
    # 4. 计算投影面积 (UTM坐标系)
    isochrones_utm = calculate_area(isochrones_gdf)
    
    # 5. 分析服务盲区
    study_area, blind_zones, total_area_m2 = analyze_blind_zone(subway_coords, radius, isochrones_gdf)
    
    # 6. 生成可视化地图
    plot_maps(G, subway_coords, isochrones_gdf, study_area, blind_zones, output_dir)
    
    # 7. 输出统计结果
    print_statistics(isochrones_utm, total_area_m2)
    
    print(f"\n分析完成！所有文件已保存到: {output_dir}")


if __name__ == "__main__":
    main()

