import socket
import threading

def handle_client(client):
    client.send(b'Request')
    while True:
        data = client.recv(1024)
        if not data:
            return
        print(f'Received data from client: {data.decode("utf-8")}')

host = '127.0.0.1'
port = 8000

#Create server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))

print("Listening on 8000")

#Listen for connections
server_socket.listen()

while True:
    client, addr = server_socket.accept()

    print(f'Accepted cxn from {addr[0]}:{addr[1]}')

    client_handler = threading.Thread(target=handle_client, args=(client,))
    client_handler.start()
