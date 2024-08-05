# CloudSentinel Deployment Script

This README provides instructions for deploying the CloudSentinel application. The deployment script ensures the necessary components are installed and configured on your system.

## Prerequisites

- Docker is installed
- Minikube is installed
- Git is installed

### Steps

1. **Start Minikube**

    ```bash
    minikube start
    ```

2. **Install kubectl (if not already installed)**

    ```bash
    if ! command -v kubectl &> /dev/null; then
        curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x ./kubectl
        sudo mv ./kubectl /usr/local/bin/kubectl
    fi
    ```

3. **Ensure kubectl is Configured with Minikube**

    ```bash
    mkdir -p $HOME/.kube
    sudo minikube kubectl -- config view --raw > $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    minikube update-context
    ```

4. **Install curl (if not already installed)**

    ```bash
    if ! command -v curl &> /dev/null; then
        sudo apt-get install -y curl
    fi
    ```

5. **Install Helm (if not already installed)**

    ```bash
    if ! command -v helm &> /dev/null; then
        curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
    else
        echo "Helm is already installed"
    fi
    ```

6. **Add Prometheus Helm Repository (if not already added)**

    ```bash
    if ! helm repo list | grep -q 'prometheus-community'; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
    else
        echo "Prometheus Helm repository is already added"
    fi
    ```

7. **Deploy Prometheus (if not already deployed)**

    ```bash
    if ! kubectl get namespaces | grep -q 'monitoring'; then
        kubectl create namespace monitoring || true
    else
        echo "Namespace 'monitoring' already exists"
    fi

    if ! helm list -n monitoring | grep -q 'prometheus'; then
        helm install prometheus prometheus-community/prometheus -n monitoring
    else
        echo "Prometheus is already deployed in the 'monitoring' namespace"
    fi
    ```

8. **Pull Docker images**

    ```bash
    DOCKER_USERNAME="jojojochem"
    docker pull $DOCKER_USERNAME/data_ingestion:latest
    docker pull $DOCKER_USERNAME/data_processing:latest
    docker pull $DOCKER_USERNAME/anomaly_detection_cgnn:latest
    docker pull $DOCKER_USERNAME/anomaly_detection_crca:latest
    docker pull $DOCKER_USERNAME/learning_adaptation:latest
    docker pull $DOCKER_USERNAME/monitoring_project:latest
    ```

9. **Download Kubernetes YAML Files from GitHub**

    ```bash
    GITHUB_REPO="https://raw.githubusercontent.com/jojoJochem/CloudSentinel/main/k8s"
    curl -LO $GITHUB_REPO/namespace-deployment.yml
    curl -LO $GITHUB_REPO/redis-deployment.yml
    curl -LO $GITHUB_REPO/data_ingestion-deployment.yml
    curl -LO $GITHUB_REPO/data_ingestion_celery-deployment.yml
    curl -LO $GITHUB_REPO/data_processing-deployment.yml
    curl -LO $GITHUB_REPO/cgnn_anomaly_detection-deployment.yml
    curl -LO $GITHUB_REPO/crca_anomaly_detection-deployment.yml
    curl -LO $GITHUB_REPO/crca_anomaly_detection-celery-deployment.yml
    curl -LO $GITHUB_REPO/learning_adaptation-deployment.yml
    curl -LO $GITHUB_REPO/learning_adaptation-celery-deployment.yml
    curl -LO $GITHUB_REPO/monitoring_project-deployment.yml
    ```

10. **Deploy to Kubernetes using YAML Files**

    ```bash
    kubectl apply -f namespace-deployment.yml
    kubectl apply -f redis-deployment.yml
    kubectl apply -f data_ingestion-deployment.yml
    kubectl apply -f data_ingestion_celery-deployment.yml
    kubectl apply -f data_processing-deployment.yml
    kubectl apply -f cgnn_anomaly_detection-deployment.yml
    kubectl apply -f crca_anomaly_detection-deployment.yml
    kubectl apply -f crca_anomaly_detection-celery-deployment.yml
    kubectl apply -f learning_adaptation-deployment.yml
    kubectl apply -f learning_adaptation-celery-deployment.yml
    kubectl apply -f monitoring_project-deployment.yml
    ```

11. **Expose the Monitoring Project Service**

    ```bash
    kubectl port-forward svc/monitoring-project-service 30000:80 -n cloudsentinel
    ```

    You can now access the interface on http://localhost:30000
