import osmnx as ox
import networkx as nx
import folium
import random

def build_disaster_router():
    print("[System] Downloading street network for Hyde Park, Chicago...")
    # Download the real-world street graph (Nodes and Edges)
    place_name = "Hyde Park, Chicago, Illinois, USA"
    G = ox.graph_from_place(place_name, network_type='drive')
    
    # Pick a random Start (Fire Station) and End (Hospital/Safe Zone)
    nodes = list(G.nodes())
    start_node = nodes[10]
    end_node = nodes[-10]
    
    print("[System] Calculating standard route...")
    # Calculate the fastest route BEFORE the disaster
    route_normal = nx.shortest_path(G, start_node, end_node, weight='length')
    
    # SIMULATE THE AI DISASTER MASK
    print("[System] Simulating U-Net disaster mask (destroying roads)...")
    edges = list(G.edges(keys=True))
    destroyed_edges = random.sample(edges, k=20)
    
    # Remove the destroyed roads from our mathematical graph
    for u, v, k in destroyed_edges:
        G.remove_edge(u, v, key=k)
        
    print("[System] Calculating dynamic emergency route...")
    # Calculate the new route AROUND the disaster zone
    try:
        route_emergency = nx.shortest_path(G, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        print("CRITICAL: No valid path exists! Area is completely cut off.")
        return

    # Plot the results natively onto an interactive Folium map
    print("[System] Generating interactive map...")
    
    # Grab the GPS coordinates of the starting intersection to center the camera
    start_y = G.nodes[start_node]['y']
    start_x = G.nodes[start_node]['x']
    route_map = folium.Map(location=[start_y, start_x], zoom_start=14, tiles="cartodbdark_matter")

    # Helper function to extract node coordinates and draw a path
    def draw_route(route_nodes, color, weight, opacity=1.0):
        # Extract the (latitude, longitude) for every intersection in the route
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route_nodes]
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity).add_to(route_map)

    # Plot the normal route in RED (The doomed path)
    draw_route(route_normal, color='red', weight=4, opacity=0.4)
    
    # Plot the new dynamic route in NEON GREEN (The safe path)
    draw_route(route_emergency, color='#00FF00', weight=5)
    
    # Save to an HTML file you can open in your browser
    route_map.save("disaster_route.html")
    print("[Success] Map saved as 'disaster_route.html'. Open it in your browser!")

if __name__ == "__main__":
    build_disaster_router()
