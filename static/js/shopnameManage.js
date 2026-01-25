import { auth, db } from "./firebase-config.js";
import { doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

const SESSION_TYPE_KEY = "sessionType";   // "owner" | "staff"
const STAFF_CTX_KEY = "staffContext";     // JSON string
const ACTIVE_SHOP_ID_KEY = "activeShopId";
const ACTIVE_SHOP_NAME_KEY = "activeShopName";

// ----------------------------
// Navbar UI
// ----------------------------
const navbar = document.querySelector(".navbar");
const navLeft = document.querySelector(".nav-left");
const navRight = document.querySelector(".nav-right");

if (!navbar || !navLeft || !navRight) {
  console.warn("shopnameManage.js: navbar/nav-left/nav-right not found.");
}

// Shop name under welcome (left side)
const shopSpan = document.createElement("div");
shopSpan.id = "shop-name";
shopSpan.style.fontWeight = "500";
shopSpan.style.fontSize = "0.9rem";
shopSpan.style.marginTop = "4px";
navLeft?.appendChild(shopSpan);

// Create a center area in the navbar to use the “empty space”
let navCenter = document.getElementById("nav-center");
if (!navCenter && navbar && navRight) {
  navCenter = document.createElement("div");
  navCenter.id = "nav-center";
  navCenter.style.flex = "1";
  navCenter.style.minWidth = "0";              // allow ellipsis
  navCenter.style.display = "flex";
  navCenter.style.alignItems = "center";
  navCenter.style.justifyContent = "center";
  navCenter.style.padding = "0 12px";
  navbar.insertBefore(navCenter, navRight);
}

// Staff summary (compact, single line)
let staffInline = document.getElementById("staff-inline");
if (!staffInline && navCenter) {
  staffInline = document.createElement("div");
  staffInline.id = "staff-inline";
  staffInline.style.fontSize = "0.85rem";
  staffInline.style.fontWeight = "600";
  staffInline.style.whiteSpace = "nowrap";
  staffInline.style.overflow = "hidden";
  staffInline.style.textOverflow = "ellipsis";
  staffInline.style.maxWidth = "100%";
  staffInline.style.opacity = "0.95";
  navCenter.appendChild(staffInline);
}

// ----------------------------
// Helpers
// ----------------------------
function safeParse(json) {
  try { return JSON.parse(json); } catch { return null; }
}

function clearStaffContext() {
  localStorage.removeItem(STAFF_CTX_KEY);

  // legacy cleanup
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

function hideStaffUI() {
  if (staffInline) {
    staffInline.textContent = "";
    staffInline.removeAttribute("title");
  }
}

function displayStaffInline(ctx) {
  if (!staffInline) return;

  const name = ctx.name || ctx.displayName || "Staff";
  const role = ctx.roleName || "Role";
  const level = ctx.accessLevel ?? "N/A";

  // Compact single-line summary (won’t stretch navbar height)
  staffInline.textContent = `👤 Staff: ${name} — ${role} (L${level})`;

  // Full details on hover (tooltip)
  const details = [
    `Name: ${name}`,
    `Role: ${role}`,
    `Access Level: ${level}`,
    ctx.email ? `Email: ${ctx.email}` : null,
    ctx.phone ? `Phone: ${ctx.phone}` : null
  ].filter(Boolean).join(" | ");

  staffInline.title = details;
}

// ----------------------------
// Render flows
// ----------------------------
async function renderOwner(user) {
  hideStaffUI();
  localStorage.setItem(SESSION_TYPE_KEY, "owner");
  clearStaffContext(); // important: never show staff UI for owner

  const shopDocRef = doc(db, "Shops", user.uid);
  const shopSnap = await getDoc(shopDocRef);

  if (shopSnap.exists() && shopSnap.data().shopName) {
    const shopName = shopSnap.data().shopName;
    shopSpan.textContent = `Shop: ${shopName}`;
    localStorage.setItem(ACTIVE_SHOP_ID_KEY, user.uid);
    localStorage.setItem(ACTIVE_SHOP_NAME_KEY, shopName);
    return;
  }

  // Owner setup prompt only if missing shopName
  let shopName = "";
  while (!shopName || shopName.trim() === "") {
    shopName = prompt("Please enter your Shop name (e.g., 'Pembe Factory Shop'):");
    if (!shopName || shopName.trim() === "") alert("Shop name cannot be empty!");
  }

  shopName = shopName.trim();
  await setDoc(shopDocRef, { shopName }, { merge: true });

  shopSpan.textContent = `Shop: ${shopName}`;
  localStorage.setItem(ACTIVE_SHOP_ID_KEY, user.uid);
  localStorage.setItem(ACTIVE_SHOP_NAME_KEY, shopName);
}

async function renderStaff(user, ctx) {
  // Hard validation: staffContext must match the CURRENT logged-in email
  if (!ctx?.email || ctx.email.toLowerCase() !== (user.email || "").toLowerCase() || !ctx.shopId) {
    console.warn("Stale/invalid staffContext. Falling back to owner UI.");
    localStorage.setItem(SESSION_TYPE_KEY, "owner");
    clearStaffContext();
    await renderOwner(user);
    return;
  }

  shopSpan.textContent = `Shop: ${ctx.shopName || "(Shop)"}`;
  displayStaffInline(ctx);

  localStorage.setItem(ACTIVE_SHOP_ID_KEY, ctx.shopId);
  localStorage.setItem(ACTIVE_SHOP_NAME_KEY, ctx.shopName || "");
}

// ----------------------------
// Auth listener
// ----------------------------
auth.onAuthStateChanged(async (user) => {
  if (!user) {
    window.location.href = "/";
    return;
  }

  const sessionType = localStorage.getItem(SESSION_TYPE_KEY) || "owner";
  const staffCtx = safeParse(localStorage.getItem(STAFF_CTX_KEY));

  if (sessionType === "staff") {
    await renderStaff(user, staffCtx);
  } else {
    await renderOwner(user);
  }
});

console.log("✅ shopnameManage.js loaded (center staff summary)");