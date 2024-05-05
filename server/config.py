config = {
    'model_type': 0, # 0: SimpleCNN, 1: ResNet 18, 2: ResNet34, 3: ResNet50, 4: ResNet101
    'is_iid': 1,
    'num_rounds': 10,
    'num_chunks': 100,
    'num_replicas': 1,
    'server_ip': '127.0.0.1',
    'server_port': 8080,
    'synchronizer_ip': '172.20.14.53',
    'synchronizer_port': 6000,
    'num_nodes': 3 # Num Training Clients + 1
}

