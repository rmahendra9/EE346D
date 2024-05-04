import socket
import argparse
from scheduler import Optimal_Schedule
from logging import INFO 
from flwr.common.logger import log 
from config import config
import csv
import time
# We load the config from the file
CLIENTS_PATH = './clients.csv'

is_iid = config['is_iid']
model_type = config['model_type']
num_rounds = config['num_rounds']
num_nodes = config['num_nodes']
num_chunks = config['num_chunks']
num_replicas = config['num_replicas']
num_segments = num_chunks*num_replicas
import pickle
# read csv file of form
# node_id, ip, port
# and get list 
# [(ip_0,port_0),(ip_1, port_1)]

# We get the info we need from a file and pass it to the clients
# client - python client.py --node_id 0 --num_clients 3 --model_type 1 --is_iid 1
mappings = [''] * num_nodes
with open(CLIENTS_PATH, 'r') as file:
    csv_reader = csv.reader(file)
    i = False
    for row in csv_reader:
        if i:
            mappings[int(row[0])] = (row[1], int(row[2]))
        else:
            i = True

config['mappings'] = mappings

for i in range(len(mappings)):
    initialsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    initialsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sent = False
    while not sent:
        try:
            print(f'Connecting to: {(mappings[i][0], mappings[i][1])}')
            initialsocket.connect((mappings[i][0], mappings[i][1]))
            config['node_id'] = i
            data = pickle.dumps(config)                
            #Send data
            initialsocket.sendall(data)
            sent = True
            print('Finished sending data.')
        except Exception as e:
                print(e)
                time.sleep(1)
    initialsocket.close()

print('Test')
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind((socket.gethostbyname(socket.gethostname()), 6000))
serversocket.listen(num_nodes)


for i in range(num_rounds):
    scheduler = Optimal_Schedule(num_nodes, num_segments, num_chunks, num_replicas)
    total_slots = 0
    for i in range(len(scheduler.nodes_schedule)):
        for communication in scheduler.nodes_schedule[i]:
            total_slots = max(total_slots, communication['slot'])
    for i in range(total_slots):
        connections = []
        for j in range(num_nodes):
            (conn, addr) = serversocket.accept()
            connections.append(conn)
            node_id = conn.recv(1024).decode()
        for j in range(len(connections)):
            connections[j].send(str(i+1).encode())
            connections[j].close()

serversocket.close()