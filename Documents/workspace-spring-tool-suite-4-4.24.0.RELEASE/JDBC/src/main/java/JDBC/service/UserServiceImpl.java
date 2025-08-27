package JDBC.service;

import JDBC.dao.userdao;
import JDBC.dao.impl.UserDaoImpl;
import JDBC.model.User;

public class UserServiceImpl implements UserService {
	userdao userDao = new UserDaoImpl();

	@Override
	public User login(String username, String password) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public User get(String username) {
		// TODO Auto-generated method stub
		return null;
	}

}
