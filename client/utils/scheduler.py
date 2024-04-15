'''
Program to create an optimal schedule (according to OptSched) given number of nodes
and number of segments (chunk-replicas).
'''

import numpy as np

from random import shuffle

class Optimal_Schedule:

    def __init__(self, num_nodes, num_segments, num_chunks, num_replicas):

        # assert num_nodes & (num_nodes - 1) == 0
        assert num_segments == num_chunks*num_replicas

        self.num_nodes = num_nodes
        self.num_segments = num_segments
        self.num_chunks = num_chunks
        self.num_replicas = num_replicas

        self.schedule, self.segment_schedule = self.initialize_schedule(self.num_nodes, [n for n in range(num_nodes)])

        for m in range(1, num_segments):
            self.insert_segment_to_schedule(m) 

        self.find_node_schedule()


    def initialize_schedule(self, num_nodes, nodes):

        assert num_nodes == len(nodes)

        schedule = []
        segment_schedule = []

        # shuffle nodes to add randomness in node-drop effects
        if 0 in nodes:
            assert nodes[0] == 0
            client_nodes = nodes[1:]
            shuffle(client_nodes)
            nodes = [0]
            nodes.extend(client_nodes)
        else:
            shuffle(nodes)

        # for i in range(1,int(np.log2(num_nodes)+1)):
        #     matching = []
        #     segment_matching = []
        #     for j in range(int(num_nodes/(2**i))):   
        #         matching.append((nodes[int(num_nodes/(2**i)+j)], nodes[j]))
        #         segment_matching.append(0)
        #     schedule.append(matching)
        #     segment_schedule.append(segment_matching)
            
        nodes_remaining = num_nodes
        while nodes_remaining > 1:
            matching = []
            segment_matching = []
            for j in range(int(np.floor(nodes_remaining/2))):
                matching.append((nodes[nodes_remaining-1-j], nodes[j]))
                segment_matching.append(0)
            schedule.append(matching)
            segment_schedule.append(segment_matching)

            nodes_remaining = int(nodes_remaining - np.floor(nodes_remaining/2))

        return schedule, segment_schedule

    
    def insert_segment_to_schedule(self, m):

        len_schedule_old = len(self.schedule)
        insert_point = int(len_schedule_old - np.ceil(np.log2(self.num_nodes))) + 1 # number of slots for which old schedule is retained

        # Start Finding Tx and Rx for Interleave Step ##################################
        nodes_tx = set() # nodes_tx will be a list of nodes that are going to transmit in interleave step
        for matching in self.schedule[insert_point+1:]:
            nodes_tx = nodes_tx.union(set([comm[0] for comm in matching]))
        nodes_rx_c = list(nodes_tx.union(set([comm[0] for comm in self.schedule[insert_point]]))) # nodes_rx_c will be the COMPLEMENT of list of nodes 
                                                                                                  # that are going to rx in interleave step
        nodes_tx = list(nodes_tx)
        assert 0 not in nodes_tx
        assert 0 not in nodes_rx_c
        assert len(nodes_tx) <= self.num_nodes // 2

        nodes_rx = [node for node in range(self.num_nodes) if node not in nodes_rx_c] # nodes_rx will be list of nodes that rx in interleave step
        shuffle(nodes_rx)

        # For the interleave step, adjust the size of the list of transmit nodes
        # size of nodes_tx has to be exactly N // 2
        other_nodes = [node for node in range(self.num_nodes) if node not in nodes_tx and node not in nodes_rx]
        shuffle(other_nodes)
        if len(nodes_tx) < self.num_nodes // 2:
            # first make sure the server is at the end of the list so that it never gets included into nodes_tx
            nodes_rx.remove(0)
            nodes_rx.append(0)
            other_nodes.extend(nodes_rx)
            nodes_tx.extend(other_nodes[:self.num_nodes//2 - len(nodes_tx)])
            nodes_rx = other_nodes[-int(np.ceil(self.num_nodes/2)):]

            shuffle(nodes_rx)
        
        shuffle(nodes_tx)            
        assert len(nodes_tx) == self.num_nodes // 2
        assert len(nodes_rx) == np.ceil(self.num_nodes / 2)

        # End Finding Tx and Rx for Interleave Step ##################################

        # Start Designing New Schedule *************************************************

        # Retain Step
        schedule_new = self.schedule[:insert_point]
        segment_schedule_new = self.segment_schedule[:insert_point]

        # Interleave Step
        matching = []
        for tx, rx in zip(nodes_tx, nodes_rx): # only forms |nodes_tx| number of pairs
            matching.append((tx, rx))
        schedule_new.append(matching)
        segment_schedule_new.append([m]*len(matching))

        # compute list of nodes yet to tx segment m
        nodes_yet_to_tx = [node for node in range(1,self.num_nodes) if node not in nodes_tx]

        # Merge Steps
        for i in range(insert_point+1, len_schedule_old + 1):

            # get matching from old schedule (we take i-1 because we added an interleave slot to new schedule)
            matching = self.schedule[i - 1]
            segment_matching = self.segment_schedule[i - 1]

            # find nodes that are in matching of old schedule
            nodes_in_old_matching = [comm[0] for comm in matching]
            nodes_in_old_matching.extend([comm[1] for comm in matching])
            # nodes_not_in_old_matching = [node for node in range(1,N) if node not in nodes_in_old_matching]

            # find common nodes that are yet to tx segment m, and free nodes in old matching
            nodes_available = [node for node in nodes_yet_to_tx if node not in nodes_in_old_matching]
            shuffle(nodes_available)
            # perform merge operation
            for idx in range(len(nodes_available) // 2):
                tx = nodes_available[len(nodes_available) - 1 - idx]
                rx = nodes_available[idx]
                matching.append((tx,rx))
                segment_matching.append(m)

                # remove tx from nodes_yet_to_tx list
                nodes_yet_to_tx.remove(tx)

            schedule_new.append(matching)
            segment_schedule_new.append(segment_matching)

        # Final Aggregate Step
        assert len(nodes_yet_to_tx) == 1
        matching = [(nodes_yet_to_tx[0], 0)]
        segment_matching = [m]
        schedule_new.append(matching)
        segment_schedule_new.append(segment_matching)

        self.schedule = schedule_new
        self.segment_schedule = segment_schedule_new
    

    def find_node_schedule(self):

        self.nodes_schedule = {}

        for node in range(self.num_nodes):

            node_schedule = []

            for slot, matching, segment_matching in zip(range(len(self.schedule)), self.schedule, self.segment_schedule):

                for i, pair in enumerate(matching):
                    if node in pair:
                        index = pair.index(node)
                        other_node = pair[1-index]
                        segment = segment_matching[i]
                        node_schedule.append({'slot': slot, 'tx': 1-index, 'other_node': other_node, 'segment': segment})
                        break
        
            self.nodes_schedule[node] = node_schedule