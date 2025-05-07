import xmlrpc.client

# Connect to server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Input from user
n = int(input("Enter an integer to compute factorial: "))

# Remote call
result = proxy.compute_factorial(n)

print(f"Factorial of {n} is: {result}")