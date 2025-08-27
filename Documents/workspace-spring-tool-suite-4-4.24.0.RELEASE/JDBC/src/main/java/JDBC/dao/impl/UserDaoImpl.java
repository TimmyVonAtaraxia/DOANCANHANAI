package JDBC.dao.impl;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

import JDBC.DBConnection;
import JDBC.dao.userdao;
import JDBC.model.User;

public class UserDaoImpl implements userdao{
	public Connection conn = null;
	public PreparedStatement ps = null;
	public ResultSet rs = null;
	@Override
	public User get(String username) {
	String sql = "SELECT * FROM [User] WHERE username = ? ";
	try {
	conn = new DBConnection().getConnectionW();
	ps = conn.prepareStatement(sql);
	ps.setString(1, username);
	rs = ps.executeQuery();
	while (rs.next()) {
	User user = new User();
	user.setId(rs.getInt("id"));
	user.setEmail(rs.getString("email"));
	user.setUserName(rs.getString("username"));
	user.setFullName(rs.getString("fullname"));
	user.setPassWord(rs.getString("password"));
	user.setAvatar(rs.getString("avatar"));

	return user; }
	} catch (Exception e) {e.printStackTrace(); }
	return null;
	}
}
