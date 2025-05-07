def create_server(name):
    return {"name": name}

def create_load_balancer(servers):
    return {"servers": servers, "index": 0}

"""def add_server(load_balancer, server):
    load_balancer["servers"].append(server)"""

def get_next_server(load_balancer):
    servers = load_balancer["servers"]
    index = load_balancer["index"]

    # Pick the next server in round-robin fashion
    selected_server = servers[index]

    # Update index for next call
    load_balancer["index"] = (index + 1) % len(servers)

    return selected_server

def prompt_server_info(index):
    name = input(f"Enter the name of server {index}: ")
    return create_server(name)

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
