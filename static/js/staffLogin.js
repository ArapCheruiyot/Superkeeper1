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
  const googleSignInBtn = document.getElementById("google-signin-btn");

  if (!googleSignInBtn) {
    console.error("❌ Google Sign-In button not found");
    return;
  }

  // Prevent double-injection
  if (document.getElementById("staff-signin-btn")) return;

  const staffLoginBtn = document.createElement("button");
  staffLoginBtn.id = "staff-signin-btn";
  staffLoginBtn.innerHTML = `<span>Staff Log In</span>`;

  staffLoginBtn.style.cssText = `
    margin-top: 15px;
    padding: 12px 24px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    width: 100%;
    max-width: 300px;
  `;

  staffLoginBtn.addEventListener("mouseenter", () => {
    staffLoginBtn.style.transform = "translateY(-2px)";
    staffLoginBtn.style.boxShadow = "0 6px 20px rgba(102, 126, 234, 0.6)";
  });

  staffLoginBtn.addEventListener("mouseleave", () => {
    staffLoginBtn.style.transform = "translateY(0)";
    staffLoginBtn.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.4)";
  });

  staffLoginBtn.addEventListener("click", handleStaffLogin);

  googleSignInBtn.parentNode.insertBefore(staffLoginBtn, googleSignInBtn.nextSibling);
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
// 4) INIT ON PAGE LOAD
// ============================================
document.addEventListener("DOMContentLoaded", () => {
  console.log("📄 staffLogin.js loaded");

  if (window.location.pathname === "/" || window.location.pathname === "/index.html") {
    injectStaffLoginButton();
  }
});

console.log("✅ staffLogin.js module loaded");