from kubernetes import client, config
from kubernetes.client import CustomObjectsApi
from nodes_metric import get_node_capacity

def get_pod_cpu_usage_for_service(app_label):

    node_capacity = get_node_capacity()
    config.load_kube_config()
    
    v1 = client.CoreV1Api()
    metrics_api = CustomObjectsApi()

    # Define the label selector to filter by the microservice label
    label_selector = f"app={app_label}"

    # Get the list of pods with the specified app label in the default namespace
    pods = v1.list_namespaced_pod(namespace='default', label_selector=label_selector).items
    
    cpu_usages = []
    
    for pod in pods:
        pod_name = pod.metadata.name
        pod_namespace = pod.metadata.namespace
        node_name = pod.spec.node_name

        # Fetch the metrics for the pod from the metrics.k8s.io API
        try:
            pod_metrics = metrics_api.get_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=pod_namespace,
                plural="pods",
                name=pod_name
            )
            cpu_millicores = 0
            # Extract CPU usage from the pod's metrics
            cpu_usage_str = pod_metrics['containers'][0]['usage']['cpu']
            if cpu_usage_str.endswith('n'):
                cpu_millicores = int(cpu_usage_str.rstrip('n')) / 1e6  # Nanocores to millicores
            elif cpu_usage_str.endswith('u'):
                cpu_millicores = int(cpu_usage_str.rstrip('u')) / 1e3  # Microcores to millicores
            else:
                cpu_millicores = int(cpu_usage_str.rstrip('m'))  # Already in millicores
            cpu_usage_pecentage = (cpu_millicores / node_capacity[node_name]['total_cpu_millicores']) * 100
            cpu_usages.append(cpu_usage_pecentage)
        except Exception as e:
            print(f"Error fetching metrics for pod {pod_name}: {str(e)}")
    
    return cpu_usages



def get_pod_ram_usage_for_service(app_label):

    node_capacity = get_node_capacity()

    config.load_kube_config()
    
    v1 = client.CoreV1Api()
    metrics_api = CustomObjectsApi()

    
    label_selector = f"app={app_label}"
    pods = v1.list_namespaced_pod(namespace='default', label_selector=label_selector).items
    
    ram_usages_mb = []
    
    for pod in pods:
        pod_name = pod.metadata.name
        pod_namespace = pod.metadata.namespace
        node_name = pod.spec.node_name

        # Fetch the metrics for the pod from the metrics.k8s.io API
        try:
            pod_metrics = metrics_api.get_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=pod_namespace,
                plural="pods",
                name=pod_name
            )

            # Extract RAM usage from the pod's metrics
            ram_usage_str = pod_metrics['containers'][0]['usage']['memory']
            if 'Ki' in ram_usage_str:
                ram_usage_mb = int(ram_usage_str.rstrip('Ki')) / 1024  # Convert KiB to MB
            elif 'Mi' in ram_usage_str:
                ram_usage_mb = int(ram_usage_str.rstrip('Mi'))  # Already in MB
            elif 'Gi' in ram_usage_str:
                ram_usage_mb = int(ram_usage_str.rstrip('Gi')) * 1024  # Convert GiB to MB
            else:
                ram_usage_mb = 0  # Default if parsing fails

            ram_usage_percentage = (ram_usage_mb / node_capacity[node_name]['total_memory_mi']) * 100
            
            ram_usages_mb.append(ram_usage_percentage)
        except Exception as e:
            print(f"Error fetching metrics for pod {pod_name}: {str(e)}")
    
    return ram_usages_mb

if __name__ == '__main__':
    cpu = get_pod_cpu_usage_for_service("ms-1")
    print(cpu)
    ram = get_pod_ram_usage_for_service("ms-1")
    print(ram)


