import socket
import threading

host = '127.0.0.1'
port = 8001

#Create server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))

print("Listening on 8001")

#Create client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, 8000))
data = client_socket.recv(1024)
msg = data.decode('utf-8')
print(msg)

#Lock
conns = 0

def handle_client(client, conns_real, msg):
    client.send(msg.encode())
    data = client.recv(1024)
    if not data:
        return
    recv_msg = data.decode('utf-8')
    print(f'Received data from client: {recv_msg}')
    if conns_real == 3:
        client_socket.send(recv_msg.encode())

#Listen for connections
server_socket.listen()

while True:
    if conns < 3:
        client, addr = server_socket.accept()
        conns += 1

        print(f'Accepted cxn from {addr[0]}:{addr[1]}')

        client_handler = threading.Thread(target=handle_client, args=(client,conns,msg,))
        client_handler.start()
        

