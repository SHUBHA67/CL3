def create_server(name, weight):
    return {"name": name, "weight": weight, "current_weight": 0}

def create_load_balancer(servers):
    return {"servers": servers}

def add_server(load_balancer, server):
    load_balancer["servers"].append(server)

def get_next_server(load_balancer):
    servers = load_balancer["servers"]
    
    # Increase current weight of each server
    for server in servers:
        server["current_weight"] += server["weight"]
    
    # Pick the server with the highest current weight
    selected_server = max(servers, key=lambda s: s["current_weight"])
    
    # Reduce the selected server's weight by total weight of all servers
    selected_server["current_weight"] -= sum(s["weight"] for s in servers)
    
    return selected_server

def prompt_server_info(index):
    name = input(f"Enter the name of server {index}: ")
    weight = int(input(f"Enter the weight of server {index}: "))
    return create_server(name, weight)

def assign_load(load_balancer, i):
    next_server = get_next_server(load_balancer)
    print(f"Load {i} assigned to server: {next_server['name']}")

if __name__ == "__main__":
    servers = []
    num_servers = int(input("Enter the number of servers: "))
    for i in range(1, num_servers + 1):
        servers.append(prompt_server_info(i))

    lb = create_load_balancer(servers)

    num_loads = int(input("Enter the number of loads: "))

    print("\nLoad balancing result:")
    for i in range(1, num_loads + 1):
        assign_load(lb, i)

