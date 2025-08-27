package JDBC;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class DBConnection{

	private final String serverName = "localhost";
	private final String dbName = "LTWCT3";
	private final String portNumber = "1433";
	private final String instance = "";

	public Connection getConnectionW() throws Exception {
		String url = "jdbc:sqlserver://" + serverName + "\\" + instance + ":" + portNumber
				+ ";integratedSecurity=true;databaseName=" + dbName;
		if (instance == null || instance.trim().isEmpty())
			url = "jdbc:sqlserver://" + serverName + ":" + portNumber + ";integratedSecurity=true;databaseName="
					+ dbName + ";encrypt=true;trustServerCertificate=true;";
		Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver");
		return DriverManager.getConnection(url);
	}

	public static void main(String[] args) {
		try {
            // Test kết nối với Windows Authentication
            DBConnection dbConn = new DBConnection();
            Connection conn = dbConn.getConnectionW();

            // Create statement
            Statement stmt = conn.createStatement();

            // Insert 'GiaoVien'
            stmt.executeUpdate("INSERT INTO Table_1(ID, NAME, DIACHI) VALUES (2, 'Trung', 'HCM')");

            // Get data from table 'Users'
            ResultSet rs = stmt.executeQuery("SELECT * FROM User");

            // Show data
            while (rs.next()) {
                System.out.println(rs.getInt("ID") + " " + rs.getString("NAME") + " " + rs.getString("DIACHI"));
            }

            conn.close(); // close connection
        } catch (Exception ex) {
            ex.printStackTrace();
        }
	}
}