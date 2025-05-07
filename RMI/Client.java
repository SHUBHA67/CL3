package RMI;

import java.io.*;
import java.rmi.*;

public class Client {
    static String name1, name2, name3;

    public static void main(String[] args) {
        BufferedReader b = new BufferedReader(new InputStreamReader(System.in));
        int ch;

        try {
            // Connect to remote object
            DBInterface DI = (DBInterface) Naming.lookup("rmi://localhost:1030/DBServ");

            do {
                System.out.println("\n1. Send input strings\n2. Display concatenated string\n0. Exit\nEnter your choice:");
                ch = Integer.parseInt(b.readLine());

                switch (ch) {
                    case 1:
                        System.out.println("Enter first string:");
                        name1 = b.readLine();
                        System.out.println("Enter second string:");
                        name2 = b.readLine();
                        name3 = DI.input(name1, name2);
                        break;

                    case 2:
                        System.out.println("Concatenated String is: " + name3);
                        break;

                    case 0:
                        System.out.println("Exiting client...");
                        break;

                    default:
                        System.out.println("Invalid choice!");
                }

            } while (ch != 0);

        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
        }
    }
}
