// staffManager.js - Staff Management System
// Handles adding, viewing, and managing staff members within each shop

import { db } from "./firebase-config.js";
import { 
  collection, 
  doc, 
  getDoc,  // ADDED: Missing import
  getDocs, 
  setDoc, 
  deleteDoc,
  query,
  where,
  orderBy,
  serverTimestamp,
  onSnapshot
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

console.log("✅ staffManager.js loaded");

let currentShopId = null;
let staffOverlay = null;
let unsubscribeStaffListener = null;

// Access level definitions
const ACCESS_LEVELS = {
  1: { 
    name: "Basic", 
    description: "Can only sell",
    color: "#4CAF50",
    permissions: ["sell"]
  },
  2: { 
    name: "Intermediate", 
    description: "Can sell & manage stock",
    color: "#2196F3", 
    permissions: ["sell", "manage_stock"]
  },
  3: { 
    name: "Advanced", 
    description: "Can sell, manage stock & view reports",
    color: "#9C27B0",
    permissions: ["sell", "manage_stock", "view_reports"]
  },
  4: { 
    name: "Admin", 
    description: "Full access including settings",
    color: "#FF9800",
    permissions: ["sell", "manage_stock", "view_reports", "manage_staff", "settings"]
  }
};

// ======================================================
// STAFF ID GENERATION UTILITIES
// ======================================================

/**
 * Generate a unique staff ID
 * Format: staff_{shopPrefix}_{timestamp}_{random}
 * Example: staff_PEMBE_1736352000000_abc123
 */
function generateStaffId() {
  const timestamp = Date.now();
  const randomStr = Math.random().toString(36).substr(2, 6);
  
  // Create a shop prefix from current shop ID (first 6 chars)
  const shopPrefix = currentShopId ? currentShopId.substring(0, 6).toUpperCase() : 'SHOP';
  
  return `staff_${shopPrefix}_${timestamp}_${randomStr}`;
}

/**
 * Find staff member by email across all shops
 * Returns: { staffUid, shopId, staffData } or null
 */
async function findStaffByEmail(email) {
  try {
    if (!email) return null;
    
    const normalizedEmail = email.toLowerCase().trim();
    
    // Get all shops
    const shopsRef = collection(db, "Shops");
    const shopsSnapshot = await getDocs(shopsRef);
    
    for (const shopDoc of shopsSnapshot.docs) {
      const shopId = shopDoc.id;
      
      // Check staff subcollection for this email
      const staffRef = collection(db, "Shops", shopId, "staff");
      const staffQuery = query(staffRef, where("email", "==", normalizedEmail));
      const staffSnapshot = await getDocs(staffQuery);
      
      if (!staffSnapshot.empty) {
        const staffDoc = staffSnapshot.docs[0];
        return {
          staffUid: staffDoc.id,  // This is our generated staff ID
          shopId: shopId,
          staffData: staffDoc.data()
        };
      }
    }
    
    return null; // Not found in any shop
  } catch (error) {
    console.error("❌ Error finding staff by email:", error);
    return null;
  }
}

// ======================================================
// INITIALIZE - Create Settings Gear Icon
// ======================================================
document.addEventListener("DOMContentLoaded", () => {
  createSettingsGearIcon();
  initializeAuth();
});

// ======================================================
// INITIALIZE AUTH
// ======================================================
function initializeAuth() {
  const auth = getAuth();
  onAuthStateChanged(auth, (user) => {
    if (user) {
      currentShopId = user.uid;
      console.log("📍 Current Shop ID:", currentShopId);
      
      // Also store in localStorage for staff login access
      localStorage.setItem('ownerUid', user.uid);
    } else {
      console.warn("⚠️ No user logged in");
    }
  });
}

// ======================================================
// CREATE SETTINGS GEAR ICON IN NAVBAR
// ======================================================
function createSettingsGearIcon() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) {
    console.warn("⚠️ Navbar not found");
    return;
  }

  const navRight = navbar.querySelector('.nav-right');
  if (!navRight) {
    console.warn("⚠️ nav-right not found");
    return;
  }

  // Check if settings button already exists
  if (document.getElementById('settings-btn')) {
    return;
  }

  // Create settings button
  const settingsBtn = document.createElement('button');
  settingsBtn.id = 'settings-btn';
  settingsBtn.innerHTML = '⚙️';
  settingsBtn.style.cssText = `
    background: rgba(255, 255, 255, 0.1);
    border: none;
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 8px;
    font-size: 20px;
    cursor: pointer;
    margin-right: 10px;
    transition: all 0.3s ease;
  `;

  settingsBtn.onmouseover = () => {
    settingsBtn.style.background = 'rgba(255, 255, 255, 0.2)';
    settingsBtn.style.transform = 'rotate(90deg)';
  };

  settingsBtn.onmouseout = () => {
    settingsBtn.style.background = 'rgba(255, 255, 255, 0.1)';
    settingsBtn.style.transform = 'rotate(0deg)';
  };

  settingsBtn.onclick = () => {
    showSettingsMenu();
  };

  // Insert before logout button
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    navRight.insertBefore(settingsBtn, logoutBtn);
  } else {
    navRight.appendChild(settingsBtn);
  }

  console.log("✅ Settings gear icon added");
}

// ======================================================
// SHOW SETTINGS MENU
// ======================================================
function showSettingsMenu() {
  // Remove existing menu if present
  const existingMenu = document.getElementById('settings-menu-overlay');
  if (existingMenu) {
    document.body.removeChild(existingMenu);
    return;
  }

  const menuOverlay = document.createElement('div');
  menuOverlay.id = 'settings-menu-overlay';
  menuOverlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3);
    z-index: 5000;
    display: flex;
    align-items: flex-start;
    justify-content: flex-end;
    padding-top: 60px;
  `;

  menuOverlay.innerHTML = `
    <div class="settings-menu" style="
      background: white;
      border-radius: 12px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.15);
      width: 260px;
      margin-right: 10px;
      overflow: hidden;
    ">
      <div style="
        padding: 16px 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
      ">
        <span>⚙️</span>
        <span>Settings</span>
      </div>
      
      <div class="settings-menu-items" style="padding: 8px 0;">
        <button class="settings-menu-item" data-action="staff" style="
          width: 100%;
          padding: 12px 20px;
          border: none;
          background: white;
          text-align: left;
          cursor: pointer;
          font-size: 14px;
          display: flex;
          align-items: center;
          gap: 10px;
          transition: all 0.2s;
          color: #444;
        ">
          <span style="font-size: 18px;">👥</span>
          <span>Manage Staff Members</span>
        </button>
      </div>
    </div>
  `;

  document.body.appendChild(menuOverlay);

  // Add hover effects
  const menuItems = menuOverlay.querySelectorAll('.settings-menu-item');
  menuItems.forEach(item => {
    item.onmouseover = () => {
      item.style.background = '#f8f9fa';
    };
    item.onmouseout = () => {
      item.style.background = 'white';
    };

    item.onclick = () => {
      const action = item.dataset.action;
      document.body.removeChild(menuOverlay);
      
      if (action === 'staff') {
        openStaffManager();
      }
    };
  });

  // Close when clicking outside
  menuOverlay.onclick = (e) => {
    if (e.target === menuOverlay) {
      document.body.removeChild(menuOverlay);
    }
  };
}

// ======================================================
// OPEN STAFF MANAGER OVERLAY
// ======================================================
async function openStaffManager() {
  if (!currentShopId) {
    await new Promise((resolve) => {
      const auth = getAuth();
      const unsubscribe = onAuthStateChanged(auth, (user) => {
        if (user) {
          currentShopId = user.uid;
          unsubscribe();
          resolve();
        }
      });
    });
  }

  if (!currentShopId) {
    alert('Please login first');
    return;
  }

  createStaffOverlay();
  setupStaffRealtimeListener();
}

// ======================================================
// CREATE COMPACT STAFF OVERLAY
// ======================================================
function createStaffOverlay() {
  // Remove existing overlay if present
  const existing = document.getElementById('staff-manager-overlay');
  if (existing) {
    document.body.removeChild(existing);
  }

  staffOverlay = document.createElement('div');
  staffOverlay.id = 'staff-manager-overlay';
  staffOverlay.style.cssText = `
    position: fixed;
    top: 60px;
    left: 0;
    width: 100%;
    height: calc(100% - 60px);
    background: rgba(0, 0, 0, 0.8);
    z-index: 4000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    box-sizing: border-box;
    backdrop-filter: blur(5px);
  `;

  staffOverlay.innerHTML = `
    <div style="
      background: white;
      border-radius: 16px;
      width: 100%;
      max-width: 900px;
      height: 100%;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    ">
      <!-- Header - Compact -->
      <div style="
        padding: 18px 24px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        flex-shrink: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
      ">
        <div>
          <h2 style="margin: 0; font-size: 20px; font-weight: 600; display: flex; align-items: center; gap: 10px;">
            <span>👥</span>
            <span>Staff Management</span>
          </h2>
          <p style="margin: 4px 0 0; font-size: 13px; opacity: 0.9;">Add and manage your team members</p>
        </div>
        <button id="close-staff-manager" style="
          background: rgba(255,255,255,0.2);
          border: none;
          color: white;
          width: 36px;
          height: 36px;
          border-radius: 8px;
          font-size: 22px;
          cursor: pointer;
          line-height: 1;
        ">×</button>
      </div>

      <!-- Search and Add Button - Compact -->
      <div style="
        padding: 16px 24px;
        background: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
        display: flex;
        gap: 10px;
        align-items: center;
        flex-shrink: 0;
      ">
        <input type="text" 
               id="staff-search" 
               placeholder="🔍 Search staff..."
               style="
                 flex: 1;
                 padding: 10px 14px;
                 border: 1px solid #ddd;
                 border-radius: 8px;
                 font-size: 14px;
                 box-sizing: border-box;
                 background: white;
               ">
        <button id="add-staff-btn" style="
          padding: 10px 20px;
          background: linear-gradient(135deg, #2ed573, #1dd1a1);
          color: white;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          font-size: 14px;
          cursor: pointer;
          white-space: nowrap;
          display: flex;
          align-items: center;
          gap: 6px;
        ">
          <span>+</span>
          <span>Add Staff</span>
        </button>
      </div>

      <!-- Compact Staff List -->
      <div style="
        flex: 1;
        overflow-y: auto;
        padding: 0;
        min-height: 200px;
        position: relative;
      ">
        <div id="staff-list-container" style="padding: 16px;">
          <!-- Staff will be loaded here -->
          <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 200px;
            color: #999;
          ">
            <div style="font-size: 40px; margin-bottom: 12px;">⏳</div>
            <p style="margin: 0; font-size: 14px;">Loading staff members...</p>
          </div>
        </div>
      </div>

      <!-- Footer with count -->
      <div id="staff-footer" style="
        padding: 12px 24px;
        background: #f8f9fa;
        border-top: 1px solid #e9ecef;
        font-size: 13px;
        color: #666;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
      ">
        <div id="staff-count">0 staff members</div>
        <div id="staff-tips" style="font-size: 12px; color: #999;">
          Click + to add new staff
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(staffOverlay);

  // Event listeners
  document.getElementById('close-staff-manager').onclick = closeStaffManager;
  document.getElementById('add-staff-btn').onclick = showAddStaffForm;
  
  // Search functionality
  const searchInput = document.getElementById('staff-search');
  let searchTimeout;
  searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      filterStaffMembers(e.target.value.toLowerCase());
    }, 300);
  });
}

// ======================================================
// SETUP REAL-TIME STAFF LISTENER
// ======================================================
function setupStaffRealtimeListener() {
  if (!currentShopId) return;
  
  // Unsubscribe from previous listener if exists
  if (unsubscribeStaffListener) {
    unsubscribeStaffListener();
  }

  const staffRef = collection(db, "Shops", currentShopId, "staff");
  const staffQuery = query(staffRef, orderBy("createdAt", "desc"));
  
  unsubscribeStaffListener = onSnapshot(staffQuery, (snapshot) => {
    // Update staff list
    const container = document.getElementById('staff-list-container');
    if (!container) return;
    
    if (snapshot.empty) {
      container.innerHTML = `
        <div style="
          text-align: center;
          padding: 60px 20px;
          color: #999;
        ">
          <div style="font-size: 48px; margin-bottom: 16px;">👥</div>
          <p style="margin: 0 0 8px; color: #666; font-size: 15px;">No Staff Members Yet</p>
          <p style="margin: 0; font-size: 13px; color: #999;">Click "Add Staff" to get started</p>
        </div>
      `;
      
      updateStaffCount(0);
      return;
    }
    
    // Render staff members in compact format
    container.innerHTML = '';
    snapshot.forEach(doc => {
      const staff = doc.data();
      const staffCard = createStaffCard(doc.id, staff);
      container.appendChild(staffCard);
    });
    
    updateStaffCount(snapshot.size);
    
  }, (error) => {
    console.error("❌ Error in real-time listener:", error);
  });
}

// ======================================================
// CREATE COMPACT STAFF CARD
// ======================================================
function createStaffCard(staffId, staffData) {
  const card = document.createElement('div');
  card.className = 'staff-card';
  card.dataset.name = staffData.name?.toLowerCase() || '';
  card.dataset.email = staffData.email?.toLowerCase() || '';
  card.dataset.role = staffData.roleName?.toLowerCase() || '';
  
  card.style.cssText = `
    display: flex;
    align-items: center;
    padding: 12px;
    border-bottom: 1px solid #f0f0f0;
    transition: all 0.2s;
    gap: 12px;
  `;

  // Get access level color
  const accessLevel = staffData.accessLevel || 1;
  const levelInfo = ACCESS_LEVELS[accessLevel] || ACCESS_LEVELS[1];
  const roleColor = levelInfo.color;

  const formattedDate = staffData.createdAt ? 
    new Date(staffData.createdAt.seconds * 1000).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    }) : '';

  card.innerHTML = `
    <!-- Role Color Indicator -->
    <div style="
      width: 4px;
      height: 40px;
      background: ${roleColor};
      border-radius: 2px;
      flex-shrink: 0;
    "></div>

    <!-- Avatar/Initial -->
    <div style="
      width: 36px;
      height: 36px;
      background: ${roleColor}20;
      color: ${roleColor};
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      font-size: 14px;
      flex-shrink: 0;
    ">
      ${(staffData.name || 'S').charAt(0).toUpperCase()}
    </div>

    <!-- Staff Details -->
    <div style="flex: 1; min-width: 0;">
      <div style="
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 4px;
      ">
        <div style="
          font-weight: 600;
          font-size: 14px;
          color: #333;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        ">${staffData.name || 'Unnamed'}</div>
        
        <div style="
          font-size: 11px;
          color: white;
          background: ${roleColor};
          padding: 2px 6px;
          border-radius: 10px;
          white-space: nowrap;
          font-weight: 600;
        ">Level ${accessLevel}</div>
      </div>
      
      <div style="display: flex; align-items: center; gap: 12px; font-size: 12px; color: #666;">
        <div style="display: flex; align-items: center; gap: 4px;">
          <span>📧</span>
          <span style="
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 180px;
          ">${staffData.email || 'No email'}</span>
        </div>
        
        ${staffData.phone ? `
          <div style="display: flex; align-items: center; gap: 4px;">
            <span>📱</span>
            <span>${staffData.phone}</span>
          </div>
        ` : ''}
        
        <div style="
          font-size: 11px;
          color: ${roleColor};
          background: ${roleColor}15;
          padding: 2px 6px;
          border-radius: 10px;
          white-space: nowrap;
        ">${staffData.roleName || levelInfo.name}</div>
        
        ${formattedDate ? `
          <div style="
            margin-left: auto;
            color: #999;
            font-size: 11px;
          ">
            ${formattedDate}
          </div>
        ` : ''}
      </div>
    </div>

    <!-- Delete Button -->
    <button class="delete-staff-btn" 
            data-id="${staffId}"
            title="Remove ${staffData.name}"
            style="
              background: transparent;
              border: 1px solid #ff6b6b;
              color: #ff6b6b;
              width: 32px;
              height: 32px;
              border-radius: 6px;
              font-size: 14px;
              cursor: pointer;
              display: flex;
              align-items: center;
              justify-content: center;
              flex-shrink: 0;
              transition: all 0.2s;
            ">
      🗑️
    </button>
  `;

  // Add hover effect
  card.onmouseover = () => {
    card.style.background = '#f8f9fa';
    card.querySelector('.delete-staff-btn').style.background = '#ff6b6b';
    card.querySelector('.delete-staff-btn').style.color = 'white';
    card.querySelector('.delete-staff-btn').style.borderColor = '#ff6b6b';
  };
  
  card.onmouseout = () => {
    card.style.background = 'transparent';
    card.querySelector('.delete-staff-btn').style.background = 'transparent';
    card.querySelector('.delete-staff-btn').style.color = '#ff6b6b';
    card.querySelector('.delete-staff-btn').style.borderColor = '#ff6b6b';
  };

  // Event listener for delete
  card.querySelector('.delete-staff-btn').onclick = () => deleteStaff(staffId, staffData.name);

  return card;
}

// ======================================================
// FILTER STAFF MEMBERS
// ======================================================
function filterStaffMembers(searchTerm) {
  const staffCards = document.querySelectorAll('.staff-card');
  let visibleCount = 0;
  
  staffCards.forEach(card => {
    const name = card.dataset.name || '';
    const email = card.dataset.email || '';
    const role = card.dataset.role || '';
    
    const matches = name.includes(searchTerm) ||
                    email.includes(searchTerm) ||
                    role.includes(searchTerm);
    
    card.style.display = matches ? '' : 'none';
    if (matches) visibleCount++;
  });
  
  updateStaffCount(visibleCount, staffCards.length);
  
  // Show empty state if no matches
  if (visibleCount === 0 && searchTerm) {
    const container = document.getElementById('staff-list-container');
    if (container) {
      container.innerHTML = `
        <div style="
          text-align: center;
          padding: 60px 20px;
          color: #999;
        ">
          <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
          <p style="margin: 0 0 8px; color: #666; font-size: 15px;">No matching staff members</p>
          <p style="margin: 0; font-size: 13px; color: #999;">Try a different search term</p>
        </div>
      `;
    }
  }
}

// ======================================================
// UPDATE STAFF COUNT
// ======================================================
function updateStaffCount(visible = null, total = null) {
  const countElement = document.getElementById('staff-count');
  if (!countElement) return;
  
  if (visible === null) {
    countElement.textContent = '0 staff members';
  } else if (total !== null && total !== visible) {
    countElement.textContent = `${visible} of ${total} staff`;
  } else {
    countElement.textContent = `${visible} staff member${visible !== 1 ? 's' : ''}`;
  }
}

// ======================================================
// CHECK PLAN LIMIT (New Function) - UPDATED PATH
// ======================================================
async function checkPlanLimit() {
  try {
    if (!currentShopId) {
      console.log("❌ No shop ID found");
      return false;
    }
    
    // Get current staff count
    const staffRef = collection(db, "Shops", currentShopId, "staff");
    const staffSnapshot = await getDocs(staffRef);
    const currentStaffCount = staffSnapshot.size;
    
    console.log(`👥 Current staff count: ${currentStaffCount}`);
    
    // CORRECTED PATH: Shops/{shopId}/plan/default (not meta/plan)
    const planRef = doc(db, "Shops", currentShopId, "plan", "default");
    const planDoc = await getDoc(planRef);
    
    if (!planDoc.exists()) {
      console.log("⚠️ No plan found at Shops/{shopId}/plan/default, defaulting to SOLO");
      // SOLO plan allows only owner (no staff)
      if (currentStaffCount = 0) { // SOLO plan allows 0 staff
        console.log("🚫 Limit reached: No plan found or SOLO plan doesn't allow staff members");
        return false;
      }
      return true;
    }
    
    const planData = planDoc.data();
    console.log("📋 Plan data:", planData);
    
    // Check for staffLimit field (your Firestore has this)
    const staffLimit = planData.staffLimit || 0;
    const planName = planData.name || "SOLO";
    
    console.log(`📊 Plan: ${planName}, Staff Limit: ${staffLimit}, Current Staff: ${currentStaffCount}`);
    
    if (currentStaffCount >= staffLimit) {
      console.log(`🚫 Limit reached for ${planName} plan (${currentStaffCount}/${staffLimit} staff)`);
      return false;
    }
    
    console.log(`✅ Can add staff. ${staffLimit - currentStaffCount} slot(s) available`);
    return true;
    
  } catch (error) {
    console.error("❌ Error checking plan limit:", error);
    // For testing, allow anyway with warning
    console.log("⚠️ Error occurred, allowing staff addition for testing");
    return true;
  }
}

// ======================================================
// SHOW ADD STAFF FORM - UPDATED FOR PLAN CHECK
// ======================================================

// ======================================================
// SHOW ADD STAFF FORM - UPDATED FOR PLAN CHECK WITH UPGRADE TRIGGER
// ======================================================
async function showAddStaffForm() {
  console.log("🔍 Checking if can add staff...");
  
  // Check plan limit before showing form
  const canAddStaff = await checkPlanLimit();
  
  if (!canAddStaff) {
    console.log("🚫 Cannot add staff: Plan limit reached");
    
    // Get current plan name for the upgrade message
    let currentPlanName = "SOLO";
    try {
      const planRef = doc(db, "Shops", currentShopId, "plan", "default");
      const planDoc = await getDoc(planRef);
      if (planDoc.exists()) {
        currentPlanName = planDoc.data().name || "SOLO";
      }
    } catch (error) {
      console.error("❌ Error getting plan name:", error);
    }
    
    // Instead of just showing toast, trigger upgrade modal
    if (typeof window.showUpgradeModal === 'function') {
      // Directly call the upgrade modal function
      window.showUpgradeModal(currentPlanName, "Add staff members");
    } else {
      // Fallback: show toast and log error
      showToast('Staff limit reached. Upgrade required to add more staff.', 'error');
      console.error('⚠️ upgrade.js not loaded or showUpgradeModal function not available');
    }
    return;
  }
  
  console.log("✅ Can add staff, showing form...");
  // ... rest of the function (show the staff form)}
  
  // If allowed, show the form
  const formOverlay = document.createElement('div');
  formOverlay.id = 'staff-form-overlay';
  formOverlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    z-index: 5000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    box-sizing: border-box;
    backdrop-filter: blur(5px);
  `;

  formOverlay.innerHTML = `
    <div style="
      background: white;
      border-radius: 16px;
      width: 100%;
      max-width: 480px;
      max-height: 90vh;
      overflow-y: auto;
      box-shadow: 0 15px 50px rgba(0,0,0,0.25);
    ">
      <div style="
        padding: 20px;
        background: linear-gradient(135deg, #2ed573, #1dd1a1);
        color: white;
        border-radius: 16px 16px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
      ">
        <h2 style="margin: 0; font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
          <span>+</span>
          <span>Add Staff Member</span>
        </h2>
        <button id="close-staff-form" style="
          background: rgba(255,255,255,0.2);
          border: none;
          color: white;
          width: 32px;
          height: 32px;
          border-radius: 6px;
          font-size: 20px;
          cursor: pointer;
          line-height: 1;
        ">×</button>
      </div>

      <form id="staff-form" style="padding: 24px;">
        <div style="margin-bottom: 16px;">
          <label style="display: block; margin-bottom: 6px; color: #555; font-size: 13px; font-weight: 500;">
            Full Name *
          </label>
          <input type="text" 
                 id="staff-name" 
                 required 
                 placeholder="e.g., John Doe"
                 autocomplete="off"
                 style="
                   width: 100%;
                   padding: 10px 12px;
                   border: 1px solid #ddd;
                   border-radius: 8px;
                   font-size: 14px;
                   box-sizing: border-box;
                   transition: all 0.3s;
                 ">
        </div>

        <div style="margin-bottom: 16px;">
          <label style="display: block; margin-bottom: 6px; color: #555; font-size: 13px; font-weight: 500;">
            Email Address *
          </label>
          <input type="email" 
                 id="staff-email" 
                 required 
                 placeholder="e.g., john@example.com"
                 autocomplete="off"
                 style="
                   width: 100%;
                   padding: 10px 12px;
                   border: 1px solid #ddd;
                   border-radius: 8px;
                   font-size: 14px;
                   box-sizing: border-box;
                   transition: all 0.3s;
                 ">
          <div style="font-size: 11px; color: #777; margin-top: 4px;">
            Staff will use this email to log in with Google
          </div>
        </div>

        <div style="margin-bottom: 16px;">
          <label style="display: block; margin-bottom: 6px; color: #555; font-size: 13px; font-weight: 500;">
            Phone Number (Optional)
          </label>
          <input type="tel" 
                 id="staff-phone" 
                 placeholder="e.g., +254712345678"
                 autocomplete="off"
                 style="
                   width: 100%;
                   padding: 10px 12px;
                   border: 1px solid #ddd;
                   border-radius: 8px;
                   font-size: 14px;
                   box-sizing: border-box;
                   transition: all 0.3s;
                 ">
        </div>

        <!-- ACCESS LEVEL SELECTION - 4 LEVEL SYSTEM -->
        <div style="margin-bottom: 16px;">
          <label style="display: block; margin-bottom: 8px; color: #555; font-size: 13px; font-weight: 500;">
            Access Level *
          </label>
          
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <!-- Level 1: Basic -->
            <label style="
              display: flex;
              align-items: center;
              padding: 12px;
              border: 1px solid #e0e0e0;
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.3s;
              background: #f9f9f9;
            ">
              <input type="radio" name="access-level" value="1" checked style="margin-right: 10px;">
              <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 13px; color: #333; margin-bottom: 2px;">
                  <span style="color: #4CAF50;">●</span> Level 1: Basic
                </div>
                <div style="font-size: 12px; color: #666;">
                  Can only sell products
                </div>
              </div>
            </label>
            
            <!-- Level 2: Intermediate -->
            <label style="
              display: flex;
              align-items: center;
              padding: 12px;
              border: 1px solid #e0e0e0;
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.3s;
              background: #f9f9f9;
            ">
              <input type="radio" name="access-level" value="2" style="margin-right: 10px;">
              <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 13px; color: #333; margin-bottom: 2px;">
                  <span style="color: #2196F3;">●</span> Level 2: Intermediate
                </div>
                <div style="font-size: 12px; color: #666;">
                  Can sell & manage stock
                </div>
              </div>
            </label>
            
            <!-- Level 3: Advanced -->
            <label style="
              display: flex;
              align-items: center;
              padding: 12px;
              border: 1px solid #e0e0e0;
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.3s;
              background: #f9f9f9;
            ">
              <input type="radio" name="access-level" value="3" style="margin-right: 10px;">
              <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 13px; color: #333; margin-bottom: 2px;">
                  <span style="color: #9C27B0;">●</span> Level 3: Advanced
                </div>
                <div style="font-size: 12px; color: #666;">
                  Can sell, manage stock & view reports
                </div>
              </div>
            </label>
            
            <!-- Level 4: Admin -->
            <label style="
              display: flex;
              align-items: center;
              padding: 12px;
              border: 1px solid #e0e0e0;
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.3s;
              background: #f9f9f9;
            ">
              <input type="radio" name="access-level" value="4" style="margin-right: 10px;">
              <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 13px; color: #333; margin-bottom: 2px;">
                  <span style="color: #FF9800;">●</span> Level 4: Admin
                </div>
                <div style="font-size: 12px; color: #666;">
                  Full access including settings
                </div>
              </div>
            </label>
          </div>
        </div>

        <!-- ROLE NAME (Display name) -->
        <div style="margin-bottom: 16px;">
          <label style="display: block; margin-bottom: 6px; color: #555; font-size: 13px; font-weight: 500;">
            Role Name (For Display) *
          </label>
          <input type="text" 
                 id="staff-role-name" 
                 required 
                 placeholder="e.g., Cashier, Stock Keeper, Manager"
                 autocomplete="off"
                 style="
                   width: 100%;
                   padding: 10px 12px;
                   border: 1px solid #ddd;
                   border-radius: 8px;
                   font-size: 14px;
                   box-sizing: border-box;
                   transition: all 0.3s;
                 ">
          <div style="font-size: 11px; color: #777; margin-top: 4px;">
            This is just a display name (e.g., "Senior Cashier", "Assistant Manager")
          </div>
        </div>

        <div style="display: flex; gap: 10px; margin-top: 24px;">
          <button type="button" id="cancel-staff-form" style="
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            background: white;
            color: #666;
            border-radius: 8px;
            font-weight: 500;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
          ">Cancel</button>
          
          <button type="submit" style="
            flex: 2;
            padding: 10px;
            border: none;
            background: linear-gradient(135deg, #2ed573, #1dd1a1);
            color: white;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
          ">Add Staff Member</button>
        </div>
      </form>
    </div>
  `;

  document.body.appendChild(formOverlay);

  // Focus on name input
  setTimeout(() => {
    document.getElementById('staff-name').focus();
  }, 100);

  // Add hover effects to radio options
  const radioLabels = formOverlay.querySelectorAll('label[style*="cursor: pointer"]');
  radioLabels.forEach(label => {
    label.addEventListener('mouseover', () => {
      label.style.background = '#f0f0f0';
      label.style.borderColor = '#2ed573';
    });
    
    label.addEventListener('mouseout', () => {
      label.style.background = '#f9f9f9';
      label.style.borderColor = '#e0e0e0';
    });
    
    // When radio is selected
    const radio = label.querySelector('input[type="radio"]');
    radio.addEventListener('change', () => {
      radioLabels.forEach(l => {
        l.style.background = '#f9f9f9';
        l.style.borderColor = '#e0e0e0';
      });
      label.style.background = '#e8f5e9';
      label.style.borderColor = '#2ed573';
    });
  });

  // Add input focus effects
  const inputs = formOverlay.querySelectorAll('input[type="text"], input[type="email"], input[type="tel"]');
  inputs.forEach(input => {
    input.addEventListener('focus', () => {
      input.style.borderColor = '#2ed573';
      input.style.boxShadow = '0 0 0 3px rgba(46, 213, 115, 0.1)';
    });
    
    input.addEventListener('blur', () => {
      input.style.borderColor = '#ddd';
      input.style.boxShadow = 'none';
    });
  });

  // Event listeners
  document.getElementById('close-staff-form').onclick = () => {
    document.body.removeChild(formOverlay);
  };
  
  document.getElementById('cancel-staff-form').onclick = () => {
    document.body.removeChild(formOverlay);
  };

  document.getElementById('cancel-staff-form').onmouseover = () => {
    document.getElementById('cancel-staff-form').style.borderColor = '#ff6b6b';
    document.getElementById('cancel-staff-form').style.color = '#ff6b6b';
  };

  document.getElementById('cancel-staff-form').onmouseout = () => {
    document.getElementById('cancel-staff-form').style.borderColor = '#ddd';
    document.getElementById('cancel-staff-form').style.color = '#666';
  };

  document.getElementById('staff-form').onsubmit = async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('staff-name').value.trim();
    const email = document.getElementById('staff-email').value.trim().toLowerCase();
    const phone = document.getElementById('staff-phone').value.trim();
    const roleName = document.getElementById('staff-role-name').value.trim();
    
    // Get selected access level
    const accessLevelInput = document.querySelector('input[name="access-level"]:checked');
    const accessLevel = accessLevelInput ? parseInt(accessLevelInput.value) : 1;

    if (!name || !email || !roleName) {
      showToast('Please fill in all required fields', 'error');
      return;
    }

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      showToast('Please enter a valid email address', 'error');
      return;
    }

    const staffData = {
      name: name,
      email: email,
      phone: phone,
      roleName: roleName,
      accessLevel: accessLevel,
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp()
    };

    await saveStaffMember(staffData);
    document.body.removeChild(formOverlay);
  };
}

// ======================================================
// SAVE STAFF MEMBER TO FIREBASE
// ======================================================
async function saveStaffMember(staffData) {
  try {
    if (!currentShopId) {
      throw new Error('No shop ID found');
    }

    // Generate staff ID using our improved function
    const staffId = generateStaffId();
    
    // Save to: Shops/{shopId}/staff/{staffId}
    const staffRef = doc(db, "Shops", currentShopId, "staff", staffId);
    
    await setDoc(staffRef, staffData);

    console.log(`✅ Staff member added:`, staffId);
    
    // Show success message
    showToast(`${staffData.name} added as ${staffData.roleName} (Level ${staffData.accessLevel})`, 'success');

  } catch (error) {
    console.error("❌ Error saving staff:", error);
    showToast('Failed to save staff member. Please try again.', 'error');
  }
}

// ======================================================
// DELETE STAFF
// ======================================================
async function deleteStaff(staffId, staffName) {
  const confirmed = confirm(`Remove "${staffName}" from staff?\n\nThis cannot be undone.`);
  
  if (!confirmed) return;

  try {
    const staffRef = doc(db, "Shops", currentShopId, "staff", staffId);
    await deleteDoc(staffRef);

    console.log(`✅ Staff member deleted: ${staffId}`);
    
    showToast(`${staffName} removed`, 'success');

  } catch (error) {
    console.error("❌ Error deleting staff:", error);
    showToast('Failed to remove staff. Please try again.', 'error');
  }
}

// ======================================================
// CLOSE STAFF MANAGER
// ======================================================
function closeStaffManager() {
  if (unsubscribeStaffListener) {
    unsubscribeStaffListener();
    unsubscribeStaffListener = null;
  }
  
  if (staffOverlay && document.body.contains(staffOverlay)) {
    document.body.removeChild(staffOverlay);
    staffOverlay = null;
  }
}

// ======================================================
// SHOW TOAST NOTIFICATION
// ======================================================
function showToast(message, type = 'success') {
  // Remove existing toast
  const existingToast = document.getElementById('staff-toast');
  if (existingToast) {
    document.body.removeChild(existingToast);
  }

  const toast = document.createElement('div');
  toast.id = 'staff-toast';
  toast.style.cssText = `
    position: fixed;
    top: 70px;
    right: 20px;
    background: ${type === 'success' ? '#2ed573' : '#ff6b6b'};
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 10000;
    font-weight: 500;
    font-size: 14px;
    animation: slideIn 0.3s ease;
    max-width: 300px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
  `;
  toast.textContent = message;

  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from { transform: translateX(300px); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(300px); opacity: 0; }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
      if (document.body.contains(toast)) {
        document.body.removeChild(toast);
      }
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    }, 300);
  }, 3000);
}

// ======================================================
// ACCESS CONTROL FUNCTIONS (For use in other files)
// ======================================================

// Get user permissions based on access level
function getPermissionsForAccessLevel(accessLevel) {
  const level = ACCESS_LEVELS[accessLevel] || ACCESS_LEVELS[1];
  return level.permissions;
}

// Check if access level has specific permission
function hasPermission(accessLevel, permission) {
  const permissions = getPermissionsForAccessLevel(accessLevel);
  return permissions.includes(permission);
}

// Apply access controls to dashboard
function applyAccessControls(accessLevel) {
  const permissions = getPermissionsForAccessLevel(accessLevel);
  
  // Control Sell button
  const sellBtn = document.getElementById('sell-btn');
  if (sellBtn) {
    sellBtn.style.display = permissions.includes('sell') ? '' : 'none';
    sellBtn.disabled = !permissions.includes('sell');
  }
  
  // Control Manage Stock button
  const stockBtn = document.getElementById('manage-stock-btn');
  if (stockBtn) {
    stockBtn.style.display = permissions.includes('manage_stock') ? '' : 'none';
    stockBtn.disabled = !permissions.includes('manage_stock');
  }
  
  // Control Business Intelligence button
  const biBtn = document.getElementById('business-intelligence-btn');
  if (biBtn) {
    biBtn.style.display = permissions.includes('view_reports') ? '' : 'none';
    biBtn.disabled = !permissions.includes('view_reports');
  }
  
  // Control Settings gear
  const settingsBtn = document.getElementById('settings-btn');
  if (settingsBtn) {
    settingsBtn.style.display = permissions.includes('settings') ? '' : 'none';
  }
  
  console.log(`🔐 Applied access controls for Level ${accessLevel}:`, permissions);
}

// ======================================================
// EXPORT FUNCTIONS
// ======================================================
window.openStaffManager = openStaffManager;
window.closeStaffManager = closeStaffManager;
window.applyAccessControls = applyAccessControls;
window.hasPermission = hasPermission;
window.getPermissionsForAccessLevel = getPermissionsForAccessLevel;
window.findStaffByEmail = findStaffByEmail;
window.ACCESS_LEVELS = ACCESS_LEVELS;
window.checkPlanLimit = checkPlanLimit; // Export for testing

// Export for use in staffLogin.js
export { findStaffByEmail, ACCESS_LEVELS, getPermissionsForAccessLevel, applyAccessControls };