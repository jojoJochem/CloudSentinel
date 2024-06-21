from kubernetes import client, config as kube_config


def load_kube_config(kube_config_path):
    kube_config.load_kube_config(kube_config_path)


def list_pod_names(namespace):
    v1 = client.CoreV1Api()
    ret = v1.list_namespaced_pod(namespace, watch=False)
    pod_names = [item.metadata.name for item in ret.items]
    return pod_names


# def list_pod_names(namespace):
#     # print(namespace)
#     v1 = client.CoreV1Api()
#     ret = v1.list_namespaced_pod(namespace, watch=False)
#     # ret = v1.list_pod_for_all_namespaces(watch=False)
#     pod_names = [item.metadata.name for item in ret.items]
#     # logging.info(f"Pod names listed for namespace {namespace}")
#     # for pod_name in pod_names:
#     #     logging.info("\t"+pod_name)
#     return pod_names
