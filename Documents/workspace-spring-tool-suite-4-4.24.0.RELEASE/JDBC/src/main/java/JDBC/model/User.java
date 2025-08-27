package JDBC.model;

import java.io.Serializable;

public class User implements Serializable {
	public static final long serialVersionUID = 1L;
	public User() {
		super();
	}
	public User(int id, String email, String userName, String fullName, String passWord, String avatar) {
		super();
		this.id = id;
		this.email = email;
		this.userName = userName;
		this.fullName = fullName;
		this.passWord = passWord;
		this.avatar = avatar;
	}
	private int id;
	private String email;
	private String userName;
	private String fullName;
	private String passWord;
	private String avatar;
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public String getEmail() {
		return email;
	}
	public void setEmail(String email) {
		this.email = email;
	}
	public String getUserName() {
		return userName;
	}
	public void setUserName(String userName) {
		this.userName = userName;
	}
	public String getFullName() {
		return fullName;
	}
	public void setFullName(String fullName) {
		this.fullName = fullName;
	}
	public String getPassWord() {
		return passWord;
	}
	public void setPassWord(String passWord) {
		this.passWord = passWord;
	}
	public String getAvatar() {
		return avatar;
	}
	public void setAvatar(String avatar) {
		this.avatar = avatar;
	}
	

}
