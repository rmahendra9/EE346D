import socket

def main():
    host = '127.0.0.1'
    port = 8000

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((host, port))

    while True:
        filename = input('Input filename you want to send: ')
        try:
            f = open(filename, 'r')
            data = f.read()
            if not data:
                break
            while data:
                client_socket.send(str(data).encode())
                data = f.read()
            f.close()
        except IOError:
            print('Invalid file name')

if __name__ == '__main__':
    main()