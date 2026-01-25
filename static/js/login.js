import { auth, provider } from "./firebase-config.js";
import {
  signInWithPopup
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const googleBtn = document.getElementById("google-signin-btn");

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
  try {
    // Pre-mark as owner session (prevents stale staff UI if anything renders early)
    localStorage.setItem("sessionType", "owner");
    clearStaffSessionKeys();

    const result = await signInWithPopup(auth, provider);
    const user = result.user;

    console.log("User info:", user);

    // Important: activeShopId must exist for dashboard modules/guards
    localStorage.setItem("activeShopId", user.uid);

    // activeShopName can be filled later by shopnameManage.js after it fetches shop doc
    localStorage.removeItem("activeShopName");

    // Redirect to dashboard
    window.location.href = "/dashboard";
  } catch (error) {
    console.error("Error signing in:", error);
    alert("Login failed");
  }
});