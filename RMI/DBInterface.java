package RMI;

import java.rmi.*;

public interface DBInterface extends Remote {
    public String input(String name1, String name2) throws RemoteException;
}
