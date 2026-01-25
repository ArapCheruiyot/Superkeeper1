import { auth, provider } from "./firebase-config.js";
import { signInWithPopup } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const googleBtn = document.getElementById("google-signin-btn");

googleBtn.addEventListener("click", () => {
    signInWithPopup(auth, provider)
        .then((result) => {
            const user = result.user;
            console.log("User info:", user);
            // Redirect to Flask dashboard page after successful login
            window.location.href = "/dashboard";
        })
        .catch((error) => {
            console.error("Error signing in:", error);
            alert("Login failed");
        });
});
