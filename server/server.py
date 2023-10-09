import socket

def main():
    host = '127.0.0.1'
    port = 8000
    num_clients = int(input('Number of clients: '))
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(num_clients)
    connections = []
    for i in range(num_clients):
        conn = server_socket.accept()
        connections.append(conn)
    
    fileno = 0
    idx = 0

    for conn in connections:
        idx += 1
        data = conn[0].recv(1024).decode()
    
        if not data:
            continue
    
        filename = 'out'+str(fileno)+'.txt'
        fileno = fileno + 1
        fo = open(filename, 'w')
        while data:
            if not data:
                break
            else:
                fo.write(data)
                data = conn[0].recv(1024).decode()
        fo.close()
    
    for conn in connections:
        conn[0].close()
    

if __name__ == '__main__':
    main()