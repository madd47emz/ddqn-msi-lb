from kubernetes import client, config

def get_pod_placements_for_service(app_label):
    
    # Load Kubernetes config
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # Define the label selector to filter by the microservice label
    label_selector = "app="+app_label


    pods = v1.list_namespaced_pod(namespace='default', label_selector=label_selector).items


    pod_placements = {}


    for pod in pods:
        pod_name = pod.metadata.name
        pod_node = pod.spec.node_name 
        pod_status = pod.status.phase
        pod_placements[pod_name] = {
            'node': pod_node,
            'status': pod_status,
        }

    return pod_placements

if __name__ == '__main__':
    placements = get_pod_placements_for_service("ms-1")
    print(placements)
