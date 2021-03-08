from pyvis.network import Network
import numpy as np

# net = Network(directed=True, layout=True)
net = Network(directed=True)
y_step = 75
counter = 0
y = 0

for i in range(10):
    first_node = counter
    net.add_node(first_node, x=0, y=y, color='red')
    counter += 1
    second_level_nodes = np.random.randint(5, 10)
    # y_2 = y - y_step
    for _ in range(second_level_nodes):
        second_node = counter
        counter += 1
        net.add_node(second_node, x=200, y=y-y_step)
        net.add_edge(first_node, second_node)
        # y_3 = y_2 - y_step
        third_level_nodes = np.random.randint(1, 5)
        for _ in range(third_level_nodes):
            third_node = counter
            counter += 1
            net.add_node(third_node, x=400, y=y-y_step)
            net.add_edge(second_node, third_node)
            y += y_step
        y += y_step
    y += y_step
net.toggle_physics(False)
net.show("mygraph.html")









