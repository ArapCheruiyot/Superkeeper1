import { auth } from "./firebase-config.js";
import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const userNameSpan = document.getElementById("user-name");
const logoutBtn = document.getElementById("logout-btn");

// Display logged-in user's name
onAuthStateChanged(auth, (user) => {
    if (user) {
        userNameSpan.textContent = `Welcome, ${user.displayName}`;
    } else {
        // No user signed in, redirect to login page
        window.location.href = "/";
    }
});

// Logout function
logoutBtn.addEventListener("click", () => {
    signOut(auth)
        .then(() => {
            window.location.href = "/"; // redirect to landing page
        })
        .catch((error) => {
            console.error("Logout error:", error);
        });
});
