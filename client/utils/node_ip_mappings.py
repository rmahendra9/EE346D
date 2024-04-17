#TODO - need list of IPs to gen mappings

def generate_node_ip_mappings():
    base_port = 2048
    mappings = {}
    for i in range(num_clients):
        ports[i+1] = ['127.0.1.1', base_port]
        base_port = base_port + 1
    return mappings

def get_node_info(node_id, ip_mappings):
    return ip_mappings[node_id]