import numpy as np

from scheduler import Optimal_Schedule

# Parameters
num_nodes = 3 # nodes are numbered 0 to num_nodes-1, with node 0 being 
              # the server (i.e., the node at which the final aggregate is received)
num_chunks = 2
num_replicas = 1 # number of replicas per chunk (DON'T WORRY ABOUT THIS PARAMETER FOR NOW)

num_segments = num_chunks*num_replicas # segments \equiv chunks in the simple case of num_replicas = 1

# Optimal_Scheduler class contains several attributes and functions describing the schedule.
# The attribute of interest for you is nodes_schedule, which I will explain a little bit below.
scheduler = Optimal_Schedule(num_nodes, num_segments, num_chunks, num_replicas)

# nodes_schedule is a dictionary, where the keys are node indices (each node has its schedule)
# in the next line, I am printing the schedule of node 1
for i in range(len(scheduler.nodes_schedule)):
    print(f' Node {i}\'s Schedule: {scheduler.nodes_schedule[i]}')

# Each node's schedule (like the one printed above) is a (ordered) list of communications that it needs to make.
# I am printing the 2nd communication that Node 1 needs to make (note: index starts from 0)
#print(f'Node 1\'s 2nd communication: {scheduler.nodes_schedule[1][1]}')

# Notice each communication variable has 4 attributes:
# slot: you can ignore this as it is only relevant for synchronous communication
# tx: if tx=1, it means this node is the transmitter in this communication; if tx=0, it is the receiver
# other_node: the other node that this node is communicating with in this communication
# segment: the chunk index that is to be communicated (chunk indices run from 0 to num_segments-1)

''' Implementation Details: (My suggestion for implementation)

* When you do chunking, implement single-threaded communication (i.e., a 
    node communications with at most one other node at a time)

* Each node is given its nodes_schedule. Below is an example schedule for Node 1:
{'slot': 0, 'tx': 1, 'other_node': 2, 'segment': 0}, 
{'slot': 1, 'tx': 0, 'other_node': 5, 'segment': 1}, 
{'slot': 2, 'tx': 1, 'other_node': 6, 'segment': 1}, 
{'slot': 3, 'tx': 0, 'other_node': 2, 'segment': 2}, 
{'slot': 4, 'tx': 1, 'other_node': 5, 'segment': 2}]

Here is how Node 1 should implement its schedule (for the schedule in the above example):
* At the start, Node 1 is "ready" to transmit chunk(segment) 0 to Node 2.
* If Node 2 is also "ready" to receive chunk 0 from Node 1, then the communication starts.
* After this communication is done, Node 1 is "ready" to receive from Node 5. If Node 5 is not "ready" to 
    to transmit to Node 1, Node 1 waits until Node 5 is ready. And then the communication starts.
* This procedure continues until all the communications in the schedule are complete.

All other nodes also follow the same protocol.
'''
