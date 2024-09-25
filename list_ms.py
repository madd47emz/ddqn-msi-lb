from kubernetes import client, config

def get_services_with_node_ports():

    config.load_kube_config()
    v1 = client.CoreV1Api()
    services = v1.list_namespaced_service(namespace='default').items

    services_with_nodeports = {}


    for service in services:
        service_name = service.metadata.name
        ports = service.spec.ports


        if service.spec.type == 'NodePort':
            node_ports = ports[0].node_port
            app_label = service.spec.selector['app']
            services_with_nodeports[service_name] = {
                'node_ports': node_ports,
                'app_label': app_label
            }

    return services_with_nodeports

# Example usage
if __name__ == "__main__":
    services = get_services_with_node_ports()
    print(services)
