package RMI;

import java.rmi.*;
import java.rmi.server.*;
import java.rmi.registry.*;

public class Server extends UnicastRemoteObject implements DBInterface {

    public Server() throws RemoteException {
        System.out.println("Initializing Server\nServer Ready");
    }

    public static void main(String[] args) {
        try {
            // Automatically creates the registry on port 1030
            LocateRegistry.createRegistry(1030);

            // Create server object and bind it to the registry
            Server rs = new Server();
            Naming.rebind("rmi://localhost:1030/DBServ", rs);
        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    // Implementation of the remote method
    public String input(String name1, String name2) {
        return name1.concat(name2);
    }
}
