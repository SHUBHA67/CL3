from Pyro5.api import expose, Daemon
import threading

@expose
class Joiner:
    def join(self, a, b):
        return a + b

daemon = Daemon()
uri = daemon.register(Joiner)
print("URI:", uri)

threading.Thread(target=daemon.requestLoop, daemon=True).start()

#----------------------------------------------------

from Pyro5.api import Proxy

uri = "PYRO:obj_39b2e8ad60ca4fefb2332349c230b445@localhost:53362"
  # Copy from server cell output
joiner = Proxy(uri)

# Input and call
a = input("First string: ")
b = input("Second string: ")
print("Result:", joiner.join(a, b))
