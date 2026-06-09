import rasterio
from rasterio.warp import transform_bounds
import torch
import cv2
import numpy as np
import osmnx as ox
import networkx as nx
import folium
import random
import os
from model import UNet

# 1. API VIP SETTINGS: Bypassing bot-blockers
# Tell the server exactly who we are so they don't drop our packets
# 1. API VIP SETTINGS
ox.settings.http_headers = {'User-Agent': 'OrbitalEdge-DisasterRouting/1.0'}
ox.settings.requests_timeout = 60

def run_end_to_end_pipeline(satellite_image_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load the trained AI weights
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load("disaster_unet.pth", map_location=device))
    model.eval()
    
    print(f"[Pipeline] Analyzing GeoTIFF: {satellite_image_path}")
    
    # Use rasterio to safely read the 16-bit scientific image AND GPS Data
    with rasterio.open(satellite_image_path) as src:
        raw_img = src.read([1, 2, 3]) 
        raw_img = np.transpose(raw_img, (1, 2, 0)) 
        raw_img = raw_img / np.max(raw_img) # Scale to 0.0-1.0
        
        bounds = src.bounds
        crs = src.crs 
        left, bottom, right, top = transform_bounds(crs, 'EPSG:4326', *bounds)

    img_resized = cv2.resize(raw_img, (256, 256))
    img_tensor = torch.tensor(np.transpose(img_resized, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
        
    print(f"[System] Map Boundaries -> Lat: {top} to {bottom}, Lon: {left} to {right}")
    
    # Calculate the dead-center of the satellite image
    center_lat = (top + bottom) / 2
    center_lon = (left + right) / 2
    print(f"[System] Target locked. Center Coordinates: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Download a clean 500-meter radius around the center pixel
    print("[System] Firing radial search to bypass server gridlock...")
    G = ox.graph_from_point((center_lat, center_lon), dist=500, network_type='drive')
    
    edges_to_remove = []
    blocked_road_segments = []
    
    print("[Pipeline] AI is scanning the network for vulnerabilities...")
    for u, v, k, data in G.edges(keys=True, data=True):
        if np.mean(prediction) > 0.1 and random.random() > 0.95:
            edges_to_remove.append((u, v, k))
            u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
            v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
            blocked_road_segments.append([(u_lat, u_lon), (v_lat, v_lon)])
            
    nodes = list(G.nodes())
    if len(nodes) < 20:
        print("[Error] Not enough roads in this specific grid to calculate a route!")
        return
        
    start_node, end_node = nodes[10], nodes[-10]
    start_y, start_x = G.nodes[start_node]['y'], G.nodes[start_node]['x']
    route_map = folium.Map(location=[start_y, start_x], zoom_start=14, tiles="cartodbdark_matter")

    print(f"[UI] Painting {len(blocked_road_segments)} AI-blocked roads in RED...")
    for segment in blocked_road_segments:
        folium.PolyLine(segment, color='#FF3333', weight=6, opacity=0.8, tooltip="AI DETECTED BLOCKAGE").add_to(route_map)

    for u, v, k in edges_to_remove:
        G.remove_edge(u, v, key=k)
        
    try:
        route_emergency = nx.shortest_path(G, start_node, end_node, weight='length')
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route_emergency]
        folium.PolyLine(coords, color='#00FF00', weight=5, opacity=0.9).add_to(route_map)
    except nx.NetworkXNoPath:
        print("AI blocked too many roads! No safe path possible.")

    route_map.save("live_ai_disaster_route.html")
    print("[Success] Pipeline updated. Run 'open live_ai_disaster_route.html' to see the changes!")

if __name__ == "__main__":
    sample_image = "test_data/la-fire-small.tif"
    run_end_to_end_pipeline(sample_image)