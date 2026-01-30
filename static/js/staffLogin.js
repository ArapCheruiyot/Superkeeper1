import { auth, db } from "./firebase-config.js";
import {
  signInWithPopup,
  GoogleAuthProvider,
  signOut
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

import {
  collectionGroup,
  query,
  where,
  getDocs,
  doc,
  getDoc
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

// ----------------------------
// Storage keys (clean separation)
// ----------------------------
const SESSION_TYPE_KEY = "sessionType";     // "owner" | "staff"
const STAFF_CTX_KEY = "staffContext";       // JSON
const ACTIVE_SHOP_ID_KEY = "activeShopId";
const ACTIVE_SHOP_NAME_KEY = "activeShopName";

// ----------------------------
// Helpers
// ----------------------------
function clearStaffSession() {
  localStorage.removeItem(STAFF_CTX_KEY);

  // Optional legacy cleanup (from your old attempt)
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

function setStaffSession(staffCtx) {
  localStorage.setItem(SESSION_TYPE_KEY, "staff");
  localStorage.setItem(STAFF_CTX_KEY, JSON.stringify(staffCtx));
  localStorage.setItem(ACTIVE_SHOP_ID_KEY, staffCtx.shopId);
  localStorage.setItem(ACTIVE_SHOP_NAME_KEY, staffCtx.shopName || "");
}

// ============================================
// 1) INJECT STAFF LOGIN BUTTON ON HOME PAGE
// ============================================
function injectStaffLoginButton() {
  console.log("🔍 staffLogin.js: Looking for Google button...");
  
  const googleSignInBtn = document.getElementById("google-signin-btn");

  if (!googleSignInBtn) {
    console.error("❌ Google Sign-In button not found");
    return;
  }

  console.log("✅ staffLogin.js: Google button found:", googleSignInBtn);

  // Prevent double-injection
  if (document.getElementById("staff-signin-btn")) {
    console.log("ℹ️ Staff button already exists");
    return;
  }

  const staffLoginBtn = document.createElement("button");
  staffLoginBtn.id = "staff-signin-btn";
  staffLoginBtn.innerHTML = `Staff Log In`;
  staffLoginBtn.className = "btn-staff"; // Add a class for easier styling

  // SIMPLER STYLING - will be side by side with Google button
  staffLoginBtn.style.cssText = `
    display: inline-block;
    margin-left: 10px;
    padding: 10px 20px;
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    vertical-align: middle;
  `;

  staffLoginBtn.addEventListener("mouseenter", () => {
    staffLoginBtn.style.transform = "translateY(-2px)";
    staffLoginBtn.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.5)";
  });

  staffLoginBtn.addEventListener("mouseleave", () => {
    staffLoginBtn.style.transform = "translateY(0)";
    staffLoginBtn.style.boxShadow = "0 2px 10px rgba(102, 126, 234, 0.3)";
  });

  staffLoginBtn.addEventListener("click", handleStaffLogin);

  // Insert it next to the Google button
  googleSignInBtn.parentNode.appendChild(staffLoginBtn);
  console.log("✅ Staff Login button injected successfully");
}

// ============================================
// 2) HANDLE STAFF LOGIN PROCESS
// ============================================
async function handleStaffLogin() {
  console.log("🔐 Staff login initiated...");

  const provider = new GoogleAuthProvider();
  // Optional: forces account chooser each time
  provider.setCustomParameters({ prompt: "select_account" });

  try {
    // IMPORTANT: clear old staff session before starting a new one
    clearStaffSession();

    const result = await signInWithPopup(auth, provider);
    const user = result.user;

    const email = (user?.email || "").trim().toLowerCase();
    if (!email) {
      await signOut(auth);
      alert("Login failed: Google did not return an email.");
      return;
    }

    console.log("👤 User signed in:", email);

    // 3) Verify user is staff
    const staffData = await findStaffByEmail(email);

    if (!staffData) {
      console.log("❌ Email not found in staff database");
      alert("Access Denied: You are not registered as a staff member. Please contact your shop owner.");

      await signOut(auth);
      clearStaffSession();
      return;
    }

    console.log("✅ Staff verified:", staffData);

    // 4) Store staff context (ONLY staff keys; do not write owner keys)
    const staffContext = {
      // auth
      uid: user.uid,
      email,

      // staff
      staffId: staffData.staffId,
      name: staffData.name || user.displayName || "",
      phone: staffData.phone || "",
      roleName: staffData.roleName || "",
      accessLevel: staffData.accessLevel ?? null,

      // shop
      shopId: staffData.shopId,
      shopName: staffData.shopName || "",

      // optional metadata
      staffDocPath: staffData.staffDocPath || ""
    };

    setStaffSession(staffContext);

    console.log("💾 staffContext saved:", staffContext);

    // 5) Redirect
    window.location.href = "/dashboard";
  } catch (error) {
    console.error("❌ Staff login error:", error);

    if (error?.code === "auth/popup-closed-by-user") return;

    alert("Login failed: " + (error?.message || "Unknown error"));
  }
}

// ============================================
// 3) FIND STAFF MEMBER (fast: collectionGroup)
// ============================================
async function findStaffByEmail(email) {
  console.log("🔍 Searching staff via collectionGroup for:", email);

  // NOTE: this requires staff docs to be under Shops/{shopId}/staff/{staffId}
  const q = query(collectionGroup(db, "staff"), where("email", "==", email));
  const snap = await getDocs(q);

  if (snap.empty) return null;

  // If an email exists under multiple shops, we use the first match
  const staffDoc = snap.docs[0];
  const staff = staffDoc.data();

  // staffDoc.ref.path => Shops/{shopId}/staff/{staffId}
  const shopId = staffDoc.ref.parent.parent.id;
  const staffId = staffDoc.id;

  // Prefer shop document name, fallback to staff.shopName
  let shopName = staff.shopName || "Unknown Shop";
  try {
    const shopSnap = await getDoc(doc(db, "Shops", shopId));
    if (shopSnap.exists()) {
      const shopData = shopSnap.data();
      shopName = shopData.shopName || shopData.name || shopName;
    }
  } catch (e) {
    // ignore, fallback already set
  }

  return {
    shopId,
    shopName,
    staffId,
    staffDocPath: staffDoc.ref.path,
    email: staff.email,
    name: staff.name,
    phone: staff.phone,
    roleName: staff.roleName,
    accessLevel: staff.accessLevel,
    createdAt: staff.createdAt,
    updatedAt: staff.updatedAt
  };
}

// ============================================
// 4) MULTIPLE INIT METHODS FOR RELIABILITY
// ============================================

// Method 1: Run when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  console.log("📄 staffLogin.js: DOMContentLoaded");
  initStaffLogin();
});

// Method 2: Run when window loads
window.addEventListener("load", () => {
  console.log("📄 staffLogin.js: window.load");
  initStaffLogin();
});

// Method 3: Run immediately if DOM is already loaded
if (document.readyState === 'loading') {
  console.log("📄 staffLogin.js: Document still loading");
} else {
  console.log("📄 staffLogin.js: Document already ready, running init");
  initStaffLogin();
}

// Method 4: Try again after a short delay (in case other scripts interfere)
setTimeout(() => {
  console.log("📄 staffLogin.js: Delayed init check");
  if (!document.getElementById("staff-signin-btn")) {
    initStaffLogin();
  }
}, 1000);

function initStaffLogin() {
  console.log("🚀 staffLogin.js: Initializing...");
  
  // Only inject on homepage
  if (window.location.pathname === "/" || 
      window.location.pathname === "/index.html" ||
      window.location.pathname === "") {
    console.log("🏠 staffLogin.js: On homepage, injecting button...");
    injectStaffLoginButton();
  } else {
    console.log("🏠 staffLogin.js: Not on homepage, skipping");
  }
}

console.log("✅ staffLogin.js module loaded");