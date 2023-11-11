import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import io

# Define the functions for graph operations as you have provided


def create_graph(edges):
    """
    Creates and returns a graph from a list of edges.
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def visualize_graph(G):
    """
    Generates a matplotlib plot of the graph and converts it to an image.
    """
    pos = nx.spring_layout(G)  # positions for all nodes
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


def increment_node(G, state, node, threshold=4):
    """
    Increment the value of the node and its connected nodes by 1,
    resetting them to 0 if they reach the threshold.
    """
    nodes_to_increment = [node] + list(G[node])
    for n in nodes_to_increment:
        state[n] = (state[n] + 1) % threshold
    return state


# Bruteforce solution
def find_solution(graph, target_state):
    """Find a sequence of node increments that results in the target state."""
    # Initial state of the graph
    initial_state = {node: 0 for node in graph}

    # We'll use a queue to keep track of all states we need to visit
    from collections import deque

    queue = deque([(initial_state, [])])

    # Set to keep track of visited states to prevent revisiting
    visited = set()

    while queue:
        current_state, sequence = queue.popleft()

        # Convert state to a tuple so it can be added to the set
        state_tuple = tuple(current_state.items())
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        # Check if the current state matches the target state
        if all(
            current_state[node] == target_state.get(node, current_state[node])
            for node in current_state
        ):
            return sequence  # Return the sequence of clicks that led to this state

        # Add new states to the queue
        for node in graph:
            new_state = increment_node(graph, current_state.copy(), node)
            queue.append((new_state, sequence + [node]))

    return None  # Return None if no solution is found


# Function to save plt figure to a buffer
def save_plt_as_image(G):
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="white",
        font_size=15,
        font_weight="bold",
    )
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf


# Function to handle node button clicks
def node_button_click(node_name):
    if node_name not in st.session_state["selected_nodes"]:
        # Add the node to the selected list if it's not already selected
        st.session_state["selected_nodes"].append(node_name)
        if len(st.session_state["selected_nodes"]) == 2:
            # If two nodes are selected, create an edge and reset
            node1, node2 = st.session_state["selected_nodes"]
            st.session_state["edges"].add((node1, node2))
            st.session_state["selected_nodes"] = []


# Initialize session state for edges and nodes
if "nodes" not in st.session_state:
    st.session_state["nodes"] = set()
if "edges" not in st.session_state:
    st.session_state["edges"] = set()
if "nodes" not in st.session_state:
    st.session_state["nodes"] = set()
if "edges" not in st.session_state:
    st.session_state["edges"] = set()
if "selected_nodes" not in st.session_state:
    st.session_state["selected_nodes"] = []

st.title("Node Increment Puzzle")

st.subheader("Description/Instruction")

st.write(
    "Welcome to the Node Increment Puzzle, a mind-bending game that challenges your problem-solving skills and logical thinking!"
)

st.subheader("How to Play:")

st.markdown(
    "- You are presented with a network of interconnected nodes, each labeled with a unique identifier."
)
st.markdown(
    "- Your objective is to find a sequence of nodes to increment so that only a specific node (or set of nodes) increases by one in value, while all other nodes either stay unchanged, or cycle through their maximum value and return to zero."
)
st.markdown(
    "- Each click on a node increases its value by 1. If a node's value reaches 4, it resets back to 0."
)
st.markdown(
    "- The twist is that clicking a node also increments the value of all directly connected nodes by 1, following the same rules."
)
st.markdown(
    "- Plan your moves carefully! The puzzle is solved when you achieve the desired configuration where your target node(s) are incremented correctly without disturbing the rest of the network."
)

st.write(
    "Can you solve the puzzle by clicking the right nodes in order? Test your skills and see if you can unravel the mystery of the Node Increment Puzzle!"
)
st.write("Happy puzzling!")

col1, col2 = st.columns(2)
with col1:
    # Start a form for the nodes input
    with st.form(key="nodes_form"):
        # Input for the number of nodes
        st.subheader("1. Choose number of nodes")
        num_nodes = st.number_input(
            "How many nodes do you want?",
            min_value=0,
            value=len(st.session_state["nodes"]),
            format="%d",
        )

        # Submit button for the form
        submit_nodes = st.form_submit_button(label="Update Nodes")

    G = nx.Graph()
    G.add_nodes_from(st.session_state["nodes"])
    G.add_edges_from(st.session_state["edges"])

    # Calculate the number of nodes and split the interface into two columns
    nodes = list(G.nodes())
    mid_index = len(nodes) // 2  # Find the midpoint to split the nodes list

    if submit_nodes or nodes:
        st.subheader("2. Create edges")
        st.write("Select one node and then another one to create an edge")
        # Update the nodes based on the specified number
        st.session_state["nodes"] = {f"{chr(65+i)}" for i in range(num_nodes)}

    # Display buttons for each node
    for node in sorted(st.session_state["nodes"]):
        if st.button(f"Select {node}"):
            node_button_click(node)

with col2:
    if submit_nodes or nodes:
        st.subheader("3. Select the target state of each node")
        st.write(
            "Specify the target state for each node (leave blank for nodes you don't want to set):"
        )
        target_state = {}

        # Split col2 into two new columns
        col1, col2 = st.columns(2)

        # Fill in the first column with the first half of nodes
        with col1:
            for node in nodes[:mid_index]:
                target_state[node] = st.number_input(
                    f"Target state for node {node}:",
                    min_value=0,
                    max_value=3,
                    value=0,
                    key=f"{node}_1",
                )

        # Fill in the second column with the second half of nodes
        with col2:
            for node in nodes[mid_index:]:
                target_state[node] = st.number_input(
                    f"Target state for node {node}:",
                    min_value=0,
                    max_value=3,
                    value=0,
                    key=f"{node}_2",
                )

    # Button to find solution
    if st.button("Find Solution"):
        solution = find_solution(G, target_state)
        if solution is not None:
            st.success(f"Solution found: {solution}")
        else:
            st.error("No solution found.")

# Visualize the graph
# Create a buffer to store image data
# Create the graph using the defined edges
if G:
    graph_image = save_plt_as_image(G)
    img = Image.open(graph_image)
    st.image(img, caption="Graph Visualization", use_column_width=True)
