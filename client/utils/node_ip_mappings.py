class IP_Mapper():
    def __init__(self, ip_list, port_list, num_nodes):
        assert(len(ip_list) >= num_nodes)
        assert(len(port_list) >= num_nodes)
        self.ip_list = ip_list
        self.port_list = port_list
        self.mappings = {}
        self.num_nodes = num_nodes
    
    def generate_node_ip_mappings(self):
        for i in range(self.num_nodes):
            mappings[i] = [ip_list[i], port_list[i]]

    def get_node_info(self, node_id):
        return self.mappings[node_id]