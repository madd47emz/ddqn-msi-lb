from kubernetes import client, config

def get_node_capacity():

    config.load_kube_config()
    v1 = client.CoreV1Api()
    
    node_capacity_data = {}
    nodes = v1.list_node().items
    
    for node in nodes:
        # Exclude master/control-plane nodes
        if 'node-role.kubernetes.io/master' in node.metadata.labels:
            continue
        node_name = node.metadata.name

        #Getting the node IP
        node_ip = None
        for addr in node.status.addresses:
            if addr.type == 'InternalIP':
                node_ip = addr.address
        
        # Total CPU in cores (convert to millicores by multiplying by 1000)
        total_cpu_millicores = int(node.status.capacity['cpu']) * 1000
        
        # Total memory in Ki (convert to Mi by dividing by 1024)
        total_memory_mi = int(node.status.capacity['memory'].rstrip('Ki')) / 1024
        
        node_capacity_data[node_name] = {
            'node_ip': node_ip,
            'total_cpu_millicores': total_cpu_millicores,
            'total_memory_mi': total_memory_mi
        }
    
    return node_capacity_data

def get_node_metrics():


    config.load_kube_config()
    api = client.CustomObjectsApi()

    # Fetch node metrics from the metrics.k8s.io API
    node_metrics = api.list_cluster_custom_object(
        group="metrics.k8s.io", 
        version="v1beta1", 
        plural="nodes"
    )

    # Get total capacity and IP of each node
    node_capacity_data = get_node_capacity()

    node_metrics_data = {}

    for item in node_metrics['items']:
        node_name = item['metadata']['name']

        # Skip master/control-plane nodes by checking if they exist in the node capacity data
        if node_name not in node_capacity_data:
            continue
        node_name = item['metadata']['name']
        usage_cpu = item['usage']['cpu']
        usage_memory = item['usage']['memory']

        if usage_cpu.endswith('n'):
            cpu_millicores = int(usage_cpu.rstrip('n')) / 1e6  # Nanocores to millicores
        elif usage_cpu.endswith('u'):
            cpu_millicores = int(usage_cpu.rstrip('u')) / 1e3  # Microcores to millicores
        else:
            cpu_millicores = int(usage_cpu.rstrip('m'))  # Already in millicores

        # Handle memory usage suffixes
        if usage_memory.endswith('Ki'):
            memory_mi = int(usage_memory.rstrip('Ki')) / 1024  # Kibibytes to MiB
        elif usage_memory.endswith('Mi'):
            memory_mi = int(usage_memory.rstrip('Mi'))  # Already in MiB
        elif usage_memory.endswith('Gi'):
            memory_mi = int(usage_memory.rstrip('Gi')) * 1024 # Gibibytes to MiB

        # Calculate percentage usage
        total_cpu_millicores = node_capacity_data[node_name]['total_cpu_millicores']
        total_memory_mi = node_capacity_data[node_name]['total_memory_mi']

        cpu_percentage = (cpu_millicores / total_cpu_millicores) * 100
        memory_percentage = (memory_mi / total_memory_mi) * 100

        # Add the node IP to the metrics data
        node_metrics_data[node_name] = {
            'node_ip': node_capacity_data[node_name]['node_ip'],
            'cpu_percentage': cpu_percentage,
            'memory_percentage': memory_percentage
        }

    return node_metrics_data

# Example usage
if __name__ == "__main__":
    metrics = get_node_metrics()
    print(metrics)