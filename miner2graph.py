from flask import Flask, jsonify, request
from services.video_fetcher import *
from services.graph_builder import *

app = Flask(__name__)

@app.route('/api/videos', methods=['GET'])
def get_videos():
    video_data = fetch_videos()
    if video_data is not None:
        return jsonify(video_data)
    else:
        return jsonify({"error": "Failed to fetch video data"}), 500

 

@app.route('/api/analyze', methods=['GET'])
def analyze_videos():
    video_data = fetch_videos()  
    G = build_graph(video_data)
    
    if G.number_of_nodes() == 0:
        return jsonify({"error": "No video data available or graph construction failed."}), 400

    degrees, community_data = analyze_and_visualize_graph(G)
    
    
    communities_json = []
    for name, community in community_data.items():
        communities_json.append({
            "name": name,
            "members": list(community)
        })

    return jsonify({
        "degrees": {node: round(degree, 4) for node, degree in degrees.items()},
        "communities": communities_json
    })


@app.route('/api/analyze_kmeans', methods=['GET'])
def analyze_videos_kmeans():
    video_data = fetch_videos()  # Assuming there's a function to fetch video data
    G = build_graph_2(video_data)
    
    if G.number_of_nodes() == 0:
        return jsonify({"error": "No video data available or graph construction failed."}), 400

    degrees, clusters_colors = analyze_and_visualize_graph_2(G)
    
    clusters_json = []
    # Generate JSON data for each cluster by extracting nodes belonging to each cluster and including the assigned color
    for cluster_label, color in clusters_colors.items():
        cluster_members = [node for node in G.nodes if G.nodes[node]['cluster'] == cluster_label]
        clusters_json.append({
            "name": f"Cluster {cluster_label}",
            "color": color,  # Include the color in the JSON response
            "members": cluster_members
        })

    return jsonify({
        "degrees": {node: round(degree, 4) for node, degree in degrees.items()},
        "clusters": clusters_json
    })


if __name__ == '__main__':
    app.run(debug=True)
