<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="jakarta.tags.core"%><!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Đăng nhập</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<style>
    body {
        font-family: Arial, sans-serif;
        background: #f8f8f8;
    }
    .login-container {
        width: 400px;
        margin: 60px auto;
        background: #fff;
        border: 1px solid #ddd;
        padding: 30px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    h2 {
        text-align: center;
        margin-bottom: 25px;
        color: #444;
    }
    .input-group {
        display: flex;
        margin-bottom: 15px;
        border: 1px solid #ccc;
        border-radius: 3px;
        overflow: hidden;
    }
    .input-group span {
        background: #f2f2f2;
        padding: 10px;
        border-right: 1px solid #ccc;
        color: #555;
        min-width: 40px;
        text-align: center;
    }
    .input-group input {
        border: none;
        padding: 10px;
        flex: 1;
    }
    .input-group input:focus {
        outline: none;
    }
    .form-options {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        margin-bottom: 15px;
    }
    .form-options label {
        color: #333;
    }
    .form-options a {
        color: #007bff;
        text-decoration: none;
    }
    .form-options a:hover {
        text-decoration: underline;
    }
    .btn-login {
        width: 100%;
        padding: 10px;
        border: none;
        background: #0099e6;
        color: white;
        font-size: 16px;
        cursor: pointer;
        border-radius: 3px;
    }
    .btn-login:hover {
        background: #007acc;
    }
    .register-link {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    .register-link a {
        color: #007bff;
        text-decoration: none;
    }
    .register-link a:hover {
        text-decoration: underline;
    }
    .alert {
        padding: 10px;
        color: white;
        background: #e74c3c;
        text-align: center;
        margin-bottom: 15px;
        border-radius: 3px;
    }
</style>
</head>
<body>
    <div class="login-container">
        <form action="login" method="post">
            <h2>Đăng Nhập Vào Hệ Thống</h2>

            <c:if test="${alert != null}">
                <div class="alert">${alert}</div>
            </c:if>

            <!-- Username -->
            <div class="input-group">
                <span><i class="fa fa-user"></i></span>
                <input type="text" name="username" placeholder="Tài khoản">
            </div>

            <!-- Password -->
            <div class="input-group">
<span><i class="fa fa-lock"></i></span>
<input type="password" name="password" placeholder="Mật khẩu">
            </div>

            <!-- Remember me + Forgot password -->
            <div class="form-options">
                <label><input type="checkbox" name="remember"> Nhớ tôi</label>
                <a href="#">Quên mật khẩu?</a>
            </div>

            <!-- Submit button -->
            <button type="submit" class="btn-login">Đăng nhập</button>

            <!-- Register link -->
            <div class="register-link">
                Nếu bạn chưa có tài khoản trên hệ thống, thì hãy
                <a href="register.jsp">Đăng ký</a>
            </div>
        </form>
    </div>
</body>
</html>