#This is the server of FL with initial Flower
import flwr as fl
print("In ORIGINAL SERVER FULL PRECISION AGG!!!!!!!!")
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_available_clients=3,
)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=60)
    )