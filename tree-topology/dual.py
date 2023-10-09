import socket
import threading
import signal
import sys

def handle_client(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        print(f'Received data from client: {data.decode("utf-8")}')
    client_socket.close()

def shutdown_server(sig, frame):
    print("Shutting down server...")
    server.close()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_server)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 8080))
server.listen(2)

print('Listening on 8080')

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8081))
client.send(b'Connecting to server as middle')

while True:
    client, addr = server.accept()
    print('Accepted cxn from {addr[0]}:{addr[1]}')

    client_handler = threading.Thread(target=handle_client, args=(client,))
    client_handler.start()


