import { auth, provider } from "./firebase-config.js";
import {
  signInWithPopup
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

// WAIT FOR PAGE TO LOAD
document.addEventListener("DOMContentLoaded", function() {
    console.log("🔍 Looking for Google button...");
    
    const googleBtn = document.getElementById("google-signin-btn");
    
    if (!googleBtn) {
        console.error("❌ Google button NOT FOUND!");
        return;
    }
    
    console.log("✅ Google button found!");
    
    function clearStaffSessionKeys() {
        // New staff system keys
        localStorage.removeItem("staffContext");

        // Legacy keys from your old attempt (must be removed to stop contamination)
        localStorage.removeItem("isStaff");
        localStorage.removeItem("shopId");
        localStorage.removeItem("ownerUid");
        localStorage.removeItem("shopName");
        localStorage.removeItem("staffName");
        localStorage.removeItem("staffEmail");
        localStorage.removeItem("staffRole");
        localStorage.removeItem("staffAccessLevel");
        localStorage.removeItem("staffPhone");
        localStorage.removeItem("staffId");
    }

    googleBtn.addEventListener("click", async () => {
        console.log("🟢 Button clicked!");
        
        try {
            // Pre-mark as owner session
            localStorage.setItem("sessionType", "owner");
            clearStaffSessionKeys();

            console.log("🔐 Opening Google popup...");
            const result = await signInWithPopup(auth, provider);
            const user = result.user;

            console.log("✅ Signed in:", user.email);

            // Set activeShopId
            localStorage.setItem("activeShopId", user.uid);
            localStorage.removeItem("activeShopName");

            console.log("🚀 Redirecting to dashboard...");
            window.location.href = "/dashboard";
        } catch (error) {
            console.error("❌ Login error:", error);
            alert("Login failed: " + error.message);
        }
    });
});