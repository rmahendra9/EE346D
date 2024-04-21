#TODO - need list of IPs to gen mappings (ping API for IPs and ports)

def generate_node_ip_mappings(num_nodes):
    #assert num_clients <= len(ip_list)
    #assert num_clients <= len(port_list)

    #TODO - this should be replaced with API call to get IP list and port list
    ip = '127.0.1.1'
    port_list = range(3055,3055+num_nodes+1)
    mappings = {}
    for i in range(num_nodes):
        mappings[i] = [ip, port_list[i]]
    return mappings

def get_node_info(node_id, ip_mappings):
    return ip_mappings[node_id]