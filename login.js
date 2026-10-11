function validateUser() {
    let u = document.getElementById("username").value.trim();
    let p = document.getElementById("password").value.trim();

    const users = {
        "Ch.Prabhavathi": "@4211",
        "B.Kusuma Priya": "@4203",
        "A.Teja": "@2303"
    };

    if (users[u] && users[u] === p) {
        localStorage.setItem("poultryvision_user", u);
        // Redirect to interactive live disease diagnostic dashboard
        window.location.href = "dashboard.html";
    } else {
        alert("Invalid Credentials! Please click one of the team member buttons below to auto-login.");
    }
}

function quickLogin(username, password) {
    document.getElementById("username").value = username;
    document.getElementById("password").value = password;
    validateUser();
}
