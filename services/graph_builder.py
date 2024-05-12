from collections import Counter
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

from sklearn.cluster import KMeans

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['text.usetex'] = False
model = SentenceTransformer('all-MiniLM-L6-v2')
similarity_threshold = 0.5

def get_embeddings(descriptions):
    try:
        return model.encode(descriptions)
    except Exception as e:
        print(f"Error encoding descriptions: {str(e)}")
        return None
    

def build_graph(video_data):
    G = nx.Graph()
    if not video_data:
        print("No video data available.")
        return G

   
    descriptions = [video.get('description', '') if video.get('description') is not None else '' for video in video_data if video]
    embeddings = get_embeddings(descriptions)
    if embeddings is None:
        print("Failed to generate embeddings.")
        return G

    similarity_matrix = np.dot(embeddings, embeddings.T)

    for i, video_a in enumerate(video_data):
        G.add_node(video_a['id'], name=video_a.get('name', 'Unknown'), description=video_a.get('description', ''))
        for j in range(i + 1, len(video_data)):
            if similarity_matrix[i, j] > similarity_threshold:
                G.add_edge(video_a['id'], video_data[j]['id'])

    return G

stop_words = set(stopwords.words('english'))

def get_community_name(descriptions):   
    filtered_descriptions = [desc for desc in descriptions if desc]
    text = ' '.join(filtered_descriptions)
    words = re.findall(r'\b\w+\b', text.lower())

    filtered_words = [word for word in words if word not in stop_words]
    most_common = Counter(filtered_words).most_common(3)
    return ' '.join(word for word, count in most_common)

def analyze_and_visualize_graph(G):
    if G is None or G.number_of_edges() == 0:
        print("The graph has no edges or is not initialized.")
        return {}, []

    degrees = nx.degree_centrality(G)
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
    except ZeroDivisionError:
        print("Division by zero occurred in community detection due to no edges.")
        return degrees, []

   
    community_names = []
    for community in communities:
        descriptions = [G.nodes[node]['description'] for node in community]
        community_name = get_community_name(descriptions)
        community_names.append(community_name)

   
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    community_colors = {node: idx for idx, community in enumerate(communities) for node in community}
    nx.draw_networkx(G, pos, node_color=[community_colors[node] for node in G.nodes], with_labels=False, node_size=[degrees[node]*2000 for node in G.nodes])
    plt.title("Video Network Graph")
    plt.savefig("network_graph.png")

    return degrees, dict(zip(community_names, communities))



def calculate_network_metrics(G):
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = float('inf') 

    average_clustering = nx.average_clustering(G)
    

    if diameter != float('inf'):
        small_world_quotient = nx.sigma(G, niter=100, nrand=10)  
  
    metrics = {
        "Diameter": diameter,
        "Average Clustering Coefficient": average_clustering,
        "Small-World Quotient": small_world_quotient if 'small_world_quotient' in locals() else 'N/A'
    }
    return metrics



def build_graph_2(video_data, n_clusters=3):
    G = nx.Graph()
    if not video_data:
        print("No video data available.")
        return G

    descriptions = [video.get('description', '') if video.get('description') is not None else '' for video in video_data if video]
    embeddings = get_embeddings(descriptions)
    if embeddings is None:
        print("Failed to generate embeddings.")
        return G

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    for i, video in enumerate(video_data):
        G.add_node(video['id'], name=video.get('name', 'Unknown'), description=video.get('description', ''),
                   cluster=labels[i])

    return G



def analyze_and_visualize_graph_2(G):
    if G is None or G.number_of_nodes() == 0:
        print("The graph has no nodes or is not initialized.")
        return {}, []

    # Define a list of colors for the clusters (expand this list if more clusters)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']  # Add more colors if needed

    # Calculate degree centrality for sizing nodes
    degrees = nx.degree_centrality(G)

    # Create a spring layout position for consistent node placement
    pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))
    for idx, cluster_label in enumerate(set(nx.get_node_attributes(G, 'cluster').values())):
        # Get all nodes in this cluster
        cluster_nodes = [node for node in G.nodes if G.nodes[node]['cluster'] == cluster_label]
        # Draw nodes in this cluster
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_size=[degrees[node] * 2000 for node in cluster_nodes], node_color=colors[idx % len(colors)], label=f'Cluster {cluster_label}')

    # Draw all edges in a neutral color
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Optional: draw node labels
    # nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title("Video Network Graph withClustering")
    plt.legend()
    plt.savefig("simplified_kmeans_network_graph.png")
    plt.close()  # Close the plot to avoid displaying it in non-interactive environments

    return degrees, dict(zip(set(nx.get_node_attributes(G, 'cluster').values()), colors))