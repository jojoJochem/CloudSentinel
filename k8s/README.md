
To monitor CPU, memory, and other metrics on a local Kubernetes cluster with Prometheus on a Mac, follow these steps:

### Step 1: Install Docker Desktop for Mac
Ensure you have Docker Desktop installed as it includes Kubernetes. If you don’t have it installed, download and install it from [Docker's website](https://www.docker.com/products/docker-desktop).

### Step 2: Enable Kubernetes in Docker Desktop
1. Open Docker Desktop.
2. Go to **Preferences** -> **Kubernetes**.
3. Check the box to enable Kubernetes.
4. Click **Apply & Restart**.

### Step 3: Install kubectl
If you don't have `kubectl` installed, you can install it via Homebrew:
```sh
brew install kubectl
```

### Step 4: Set Up Prometheus and Grafana
You can use Helm to deploy Prometheus and Grafana on your local Kubernetes cluster. First, install Helm if you don’t have it:
```sh
brew install helm
```

Then, add the Helm stable repository:
```sh
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```
### Step 1: Clear and Recreate kubeconfig
1. **Back up your existing kubeconfig**:
    ```sh
    cp ~/.kube/config ~/.kube/config.backup
    ```
2. **Clear the existing kubeconfig**:
    ```sh
    rm ~/.kube/config
    ```
3. **Restart Minikube** to recreate the kubeconfig:
    ```sh
    minikube start
    ```



### Step 5: Deploy Prometheus
Use Helm to install Prometheus:
```sh
kubectl create namespace monitoring
helm install prometheus prometheus-community/prometheus --namespace monitoring
```

### Step 6: Deploy Grafana
Use Helm to install Grafana:
```sh
helm install grafana grafana/grafana --namespace monitoring
```

### Step 7: Access Prometheus and Grafana
#### Prometheus:
To access the Prometheus dashboard, set up port forwarding:
```sh
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
```
You can then access Prometheus at `http://localhost:9090`.

#### Grafana:
To access the Grafana dashboard, set up port forwarding:
```sh
kubectl get pods -n monitoring  # Find the Grafana pod name
kubectl port-forward -n monitoring svc/grafana 3000:80
```
You can then access Grafana at `http://localhost:3000`.

Grafana default login:
- Username: `admin`
- Password: `prom-operator`

### Step 8: Configure Grafana to Use Prometheus
1. Open Grafana in your browser (`http://localhost:3000`).
2. Go to **Configuration** -> **Data Sources** -> **Add data source**.
3. Select **Prometheus**.
4. Set the URL to `http://prometheus-server.monitoring.svc.cluster.local`.
5. Click **Save & Test**.

### Step 9: Import Dashboards
You can import existing dashboards for Kubernetes monitoring. To do this:
1. In Grafana, go to **Create** -> **Import**.
2. Enter the dashboard ID from [Grafana’s Dashboard Repository](https://grafana.com/grafana/dashboards/).
3. Click **Load**, then **Import**.

### Monitoring Metrics
With Prometheus and Grafana set up, you can now monitor various metrics, including CPU and memory usage, across your Kubernetes cluster.

### Additional Resources
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)

minikube service grafana --namespace=my-grafana
 minikube service grafana prometheus-server --namespace=monitoring



