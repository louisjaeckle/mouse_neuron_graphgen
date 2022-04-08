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

    def plot_cube(self, bounds, cube_dim):
        # 3d plots subgraph in cube given by bounds
        # bounds = [xbound,ybound,zbound], cube dim is side size of square
        # returns whether succeeded

        node_names, locs = self.get_cube(bounds, cube_dim)

        if len(node_names) == 0:
            return False

        self.plot_nodes(node_names)

    #use this to 3d plot null and empirical networks
    def plot_nodes(self,node_names,edges=None):
        #node_names - all nodes to be plotted
        #edges- list of edges. if None, will be taken from empirical network


        if edges==None:
            print('edges==none')
            edges = []
            seen_nodes = set()
            for node in node_names:
                for neighbor in self.G.neighbors(node):
                    if neighbor in seen_nodes:
                        edges.append([node, neighbor])
                seen_nodes.add(node)

        x_plots = [self.node_dict[name][0] for name in node_names]
        y_plots = [self.node_dict[name][1] for name in node_names]
        z_plots = [self.node_dict[name][2] for name in node_names]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.scatter(x_plots, y_plots, z_plots, s=30, color='blue')

        # uncomment to display node labels
        #         for node in node_names:
        #             ax.text(*self.node_dict[node],node,size=10,zorder=1,color='k')

        for node1, node2 in edges:
            plt_args = [[self.node_dict[node1][i], self.node_dict[node2][i]] for i in range(3)]
            ax.plot(*plt_args)
        plt.show()

        return True

    def plot_random_cube(self,cube_dim):
        #3d plots subnetwork in cube with random bounds
        success = False
        while not success:
            bounds = self.get_random_bounds(cube_dim)
            success = self.plot_cube(bounds,cube_dim)

    # get random bounds for a cube to parse
    def get_random_bounds(self, cube_dim):
        center = [np.random.uniform(self.lo_bound[ind], self.hi_bound[ind]) for ind in range(3)]
        return [coord - (cube_dim / 2) for coord in center]

    # parses network into cube and returns nodes in cube
    def get_cube(self, bounds, cube_dim):
        # bounds = [xbound,ybound,zbound], cube dim is side size of square
        # returns names, locations sorted by ascending z coordinates (ndarrays)
        lo_ind, hi_ind = 0, 0
        locations = np.copy(self.coordinates)
        names = np.copy(self.names)

        for ind in range(3):
            lo_ind = binary_search(locations, bounds[ind], ind)
            if lo_ind == -1:
                #                 print('no nodes in range')#no nodes left
                return np.array([]), np.array([])
            hi_ind = binary_search(locations, bounds[ind] + cube_dim, ind)
            if hi_ind <= lo_ind:
                #                 print('no nodes in range')#no nodes left
                return np.array([]), np.array([])
            locations = locations[lo_ind:hi_ind]
            names = names[lo_ind:hi_ind]
            if (ind != 2):
                reorder = locations[:, ind + 1].argsort()
                locations = locations[reorder]
                names = names[reorder]
        return names, locations

    # parse cube into degrees, adjacencies
    def parse_cube(self, names, locations, dim):
        # takes nodes and parses into (dim x dim x dim) matrix, then extracts degree structure and adjacency matrix
        # names is ndarray of node names, locations ndarray of coordinates
        # formated like output of get_subgraph (ie sorted by ascending z coordinates)
        # dim is size of cube, in terms of neurons per side

        # return flattened arrays degrees,adjacencies,name which are representations of the degree structure and adjacency matrix and a list of names in order
        #and returns name

        if len(names) < dim ** 3:
            #             print('not enough nodes to parse')
            return np.array([]), np.array([]), np.array([])

        parsed_names = []
        degrees = np.zeros(dim ** 3)
        adj_matrix = [[0 for _ in range(row_len)] for row_len in
                      range(dim ** 3)]  # 2d list indexed by [higher node index],[lower node index]
        node_indices = {}
        for bott_ind in range(0, dim ** 3, dim ** 2):
            layer_names = names[bott_ind:bott_ind + dim ** 2]
            layer_locs = locations[bott_ind:bott_ind + dim ** 2]
            reorder = layer_locs[:, 1].argsort()  # sort by y
            layer_names = layer_names[reorder]
            layer_locs = layer_locs[reorder]
            for lo_ind in range(0, dim ** 2, dim):
                row_names = layer_names[lo_ind:lo_ind + dim]
                reorder = layer_locs[lo_ind:lo_ind + dim, 0].argsort()  # sort by x
                row_names = row_names[reorder]
                for col in range(dim):
                    node_index = bott_ind + lo_ind + col
                    # get neighbors, update node_info and edges
                    for neighbor in self.G.neighbors(row_names[col]):
                        if neighbor in node_indices:
                            degrees[node_index] += 1
                            degrees[node_indices[neighbor]] += 1
                            adj_matrix[node_index][node_indices[neighbor]] = 1
                    node_indices[row_names[col]] = node_index
                    parsed_names.append(row_names[col])

        adj_matrix = [val for adj_row in adj_matrix for val in adj_row]
        return degrees, np.asarray(adj_matrix),np.asarray(parsed_names)

    # gets and parses subgraph in cube with given bounds
    def get_and_parse(self, cube_dim, parse_dim, bounds):
        # return flattened ndarrays degrees,adjacencies, which are representations of the degree structure and adjacency matrix
        names, locations = self.get_cube(bounds, cube_dim)
        degrees, adjacencies,names = self.parse_cube(names, locations, parse_dim)
        return degrees, adjacencies,np.asarray(names)

    # get flattened degrees, flattened adjacencies of a random cube
    def random_get_and_parse(self, cube_dim, parse_dim):
        # random bound generation continues until viable one found
        # cube_dim- side size of cube used in subnetwork
        # parse dim- side dimension of 3d matrix to parse subgraph into
        # return flattened ndarrays degrees,adjacencies, which are representations of the degree structure and adjacency matrix
        parsed_subgraph = None
        while True:
            bounds = self.get_random_bounds(cube_dim)
            degrees, adjacencies,names = self.get_and_parse(cube_dim, parse_dim, bounds)

            if len(degrees) != 0:
                return degrees, adjacencies,names

    def example_generator(self,cube_dim,parse_dim,batch_size=32):
        n_nodes = int(parse_dim**3)#number of nodes per cube
        while True:
            train_features = np.empty((batch_size,n_nodes))
            train_labels = np.empty((batch_size,int(n_nodes*(n_nodes-1)/2)))
            for i in range(batch_size):
                features, labels,_ = self.random_get_and_parse(cube_dim, parse_dim)
                train_features[i] = features
                train_labels[i] = labels
            yield train_features,train_labels


if __name__=='__main__':
    path_to_gml = r'C:\Users\Louis\jupyter_ntbks\csci3352_mouse_proj\mouse_retina_1.graphml'##put path to mouse_retina_1.graphml here
    #can be downloaded from https://neurodata.io/project/connectomes/

    print('reading. . .')
    graph = nx.read_graphml(path_to_gml)
    print('done')
    graph = graph.to_undirected()#github says its undirected, and from/to is arbitrary, so ig do this
    mouse_connectome = Connectome(graph)

    mouse_connectome.plot_random_cube(20)