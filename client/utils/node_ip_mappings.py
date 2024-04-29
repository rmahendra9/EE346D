#TODO - need list of IPs to gen mappings (ping API for IPs and ports)

def generate_node_ip_mappings(num_nodes):
    #assert num_clients <= len(ip_list)
    #assert num_clients <= len(port_list)

    #TODO - this should be replaced with API call to get IP list and port list
    ip_list = ['10.52.2.171', '10.52.3.58', '10.52.3.70', '10.52.3.62']
    port_list = [2048]*(num_nodes)
    mappings = {}
    for i in range(num_nodes):
        mappings[i] = [ip_list[i], port_list[i]]
    return mappings

def get_node_info(node_id, ip_mappings):
    return ip_mappings[node_id]