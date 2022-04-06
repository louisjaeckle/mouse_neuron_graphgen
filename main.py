import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import sys


def binary_search(sorted_list, target, ind):  # returns index of first value >=target
    # sorted list is assumed to contain (x,y,z) coordinates. search is done based on ind coordinate (i.e. ind = 0 means x)
    # returns len(sorted_list) if not found
    lo = 0
    hi = len(sorted_list) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if sorted_list[mid][ind] >= target:
            hi = mid
        else:
            lo = mid
    if target > sorted_list[hi][ind]:
        return len(sorted_list)
    return lo if sorted_list[lo][ind] >= target else hi


class Connectome:
    # attr
    # node_locations: maps node names to (x,y,z) coords
    # names: list of node names
    # coordinates: list of (x,y,z) coords, sorted by increasing x, with order corresponding to names
    # ie node names[i] has coordinate coordinates[i]
    def __init__(self, G):
        self.G = G

        Xs = nx.get_node_attributes(G, 'x')
        Ys = nx.get_node_attributes(G, 'y')
        Zs = nx.get_node_attributes(G, 'z')

        self.node_dict = dict()
        self.lo_bound = []  # [min x, min y, min z]
        self.hi_bound = []  # [max x, max y, max z]
        # TODO add in min/max bounds
        for node in list(G.nodes):
            try:
                if float(Xs[node]) > 80:
                    continue
                self.node_dict[node] = [int(Xs[node]), int(Ys[node]), int(Zs[node])]
                if len(self.lo_bound) == 0:
                    self.lo_bound = self.node_dict[node][:]
                    self.hi_bound = self.node_dict[node][:]
                else:
                    for ind in range(3):
                        self.lo_bound[ind] = min(self.lo_bound[ind], self.node_dict[node][ind])
                        self.hi_bound[ind] = max(self.hi_bound[ind], self.node_dict[node][ind])
            except KeyError:
                pass

        self.coordinates = []  # extract [x,y,z] nodes
        self.names = []  # extract names
        for node, coord in self.node_dict.items():
            self.coordinates.append(coord)
            self.names.append(node)
        self.coordinates = np.asarray(self.coordinates)
        self.names = np.asarray(self.names)
        reorder = self.coordinates[:, 1].argsort()
        self.coordinates = self.coordinates[reorder]
        self.names = self.names[reorder]

    def get_subgraph(self, bounds, cube_dim):  # bounds = [xbound,ybound,zbound], cube dim is side size of square
        # returns ndarray of names, ndarray of locations, sortex by ascending z coordinates
        lo_ind, hi_ind = 0, 0
        locations = self.coordinates
        names = self.names
        for ind in range(3):
            lo_ind = binary_search(locations, bounds[ind], ind)
            if lo_ind == -1:
                print('no nodes in range')  # no nodes left
                return [], []
            hi_ind = binary_search(locations, bounds[ind] + cube_dim, ind)
            if hi_ind <= lo_ind:
                print('no nodes in range')  # no nodes left
                return [], []
            locations = locations[lo_ind:hi_ind]
            names = names[lo_ind:hi_ind]
            if (ind != 2):
                reorder = locations[:, ind + 1].argsort()
                locations = locations[reorder]
                names = names[reorder]
        return names, locations

    def plot_empirical_subnetwork(self, bounds, cube_dim, points_size=30):  # inputs formatted like get_subgraph above
        # returns whether succeeded

        node_names, locs = self.get_subgraph(bounds, cube_dim)

        if len(node_names) == 0:
            return False

        seen_nodes = set()

        node_pairs = []  # list of connected [node,node] pairs

        for node in node_names:
            for neighbor in self.G.neighbors(node):
                if neighbor in seen_nodes:
                    node_pairs.append([node, neighbor])
            seen_nodes.add(node)
        x_plots = [self.node_dict[name][0] for name in node_names]
        y_plots = [self.node_dict[name][1] for name in node_names]
        z_plots = [self.node_dict[name][2] for name in node_names]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.scatter(x_plots, y_plots, z_plots, s=points_size, color='blue')

        for node1, node2 in node_pairs:
            plt_args = [[self.node_dict[node1][i], self.node_dict[node2][i]] for i in range(3)]
            ax.plot(*plt_args)
        plt.show()

        return True

    def get_random_bounds(self, cube_dim):
        # get random bounds for a cube to parse. Old method was biased, new method places center of cube randomly then gets bounds
        center = [np.random.uniform(self.lo_bound[ind], self.hi_bound[ind]) for ind in range(3)]
        return [coord - (cube_dim / 2) for coord in center]

    def plot_random_empirical_subnetwork(self, cube_dim, points_size=None):
        # returns whether succeeded
        bounds = self.get_random_bounds(cube_dim)
        if points_size == None:
            return self.plot_empirical_subnetwork(bounds, cube_dim)
        else:
            return self.plot_empirical_subnetwork(bounds, cube_dim, points_size=points_size)

    def parse_subgraph(self, names, locations, dim):
        # names is ndarray of node names, locations ndarray of coordinates
        # formated like output of get_subgraph (ie sorted by ascending z coordinates)
        # dim is size of cube, in terms of neurons per side

        # return ndarray out_subgraph where out_subgraph[i][j][k] is the name of the node in the ith layer, jth row, kth col

        if len(names) < dim ** 3:
            print('not enough nodes to parse')
            return np.array([])

        out_subgraph = []
        for bott_ind in range(0, dim ** 3, dim ** 2):  # exclusive
            out_subgraph.append([])
            layer_names = names[bott_ind:bott_ind + dim ** 2]
            layer_locs = locations[bott_ind:bott_ind + dim ** 2]
            reorder = layer_locs[:, 1].argsort()  # sort by y
            layer_names = layer_names[reorder]
            layer_locs = layer_locs[reorder]
            for lo_ind in range(0, dim ** 2, dim):
                order = layer_locs[lo_ind:lo_ind + dim, 0].argsort()  # sort by x
                out_subgraph[-1].append([layer_names[i] for i in order])

        return np.asarray(out_subgraph)

    def get_bounded_subgraph(self, cube_dim, parse_dim, bounds):
        # parses subgraph with given bounds
        names, locations = self.get_subgraph(bounds, cube_dim)
        return self.parse_subgraph(names, locations, parse_dim)

    def get_random_subgraph(self, cube_dim, parse_dim, bounds=[]):
        # randomly generates a point in the space of the network (unless bounds specified), extracts a cube, and parses that into a 3d matrix
        # random bound generation continues until
        # cube_dim- side size of cube used in subnetwork
        # parse dim- side dimension of 3d matrix to parse subgraph into
        parsed_subgraph = None
        while True:
            bounds = self.get_random_bounds(cube_dim)
            names, locations = self.get_subgraph(bounds, cube_dim)
            parsed_subgraph = self.parse_subgraph(names, locations, parse_dim)
            if np.size(parsed_subgraph) != 0:
                return parsed_subgraph

    def subgraph_generator(self, cube_dim, parse_dim):
        while True:
            yield self.get_random_subgraph(cube_dim, parse_dim)


path_to_gml = r'C:\Users\Louis\PycharmProjects\mouse_connectome\mouse_retina_1.graphml'##put path to mouse_retina_1.graphml here
#can be downloaded from https://neurodata.io/project/connectomes/

print('reading. . .')
graph = nx.read_graphml(path_to_gml)
print('done')
mouse_connectome = Connectome(graph)

success = False
while not success:
    success = mouse_connectome.plot_random_empirical_subnetwork(20)