// planManager.js - Plan Management & Upgrade System
// Shows upgrade prompt when user can't add more staff

import { db } from "./firebase-config.js";
import { 
  collection, 
  doc, 
  getDoc, 
  getDocs
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

console.log("✅ planManager.js loaded");

let currentShopId = null;
let currentPlan = null;

// ======================================================
// PLAN DEFINITIONS
// ======================================================
const PLANS = {
  SOLO: {
    id: "SOLO",
    name: "SOLO",
    staffLimit: 0,
    priceKES: 0,
    description: "Owner only - No staff access",
    nextUpgrade: "BASIC"
  },
  BASIC: {
    id: "BASIC", 
    name: "BASIC",
    staffLimit: 5,
    priceKES: 250,
    description: "Owner + up to 5 staff",
    nextUpgrade: "TEAM"
  },
  TEAM: {
    id: "TEAM",
    name: "TEAM", 
    staffLimit: 10,
    priceKES: 500,
    description: "Growing team - up to 10 staff",
    nextUpgrade: "BUSINESS"
  },
  BUSINESS: {
    id: "BUSINESS",
    name: "BUSINESS",
    staffLimit: 20,
    priceKES: 1000,
    description: "Established business - up to 20 staff",
    nextUpgrade: "ENTERPRISE"
  }
};

// ======================================================
// INITIALIZATION
// ======================================================
document.addEventListener("DOMContentLoaded", () => {
  initializeAuthAndPlan();
});

// ======================================================
// INITIALIZE AUTH & LOAD PLAN
// ======================================================
function initializeAuthAndPlan() {
  const auth = getAuth();
  onAuthStateChanged(auth, async (user) => {
    if (user) {
      currentShopId = user.uid;
      console.log("📍 Current Shop ID:", currentShopId);
      
      // Load shop's plan
      await loadShopPlan();
    } else {
      console.warn("⚠️ No user logged in");
    }
  });
}

// ======================================================
// LOAD SHOP PLAN FROM FIRESTORE
// ======================================================
async function loadShopPlan() {
  if (!currentShopId) return;
  
  try {
    const planRef = doc(db, "Shops", currentShopId, "plan", "default");
    const planDoc = await getDoc(planRef);
    
    if (planDoc.exists()) {
      currentPlan = planDoc.data();
      console.log("📋 Current Plan Loaded:", currentPlan.name, "Staff Limit:", currentPlan.staffLimit);
    } else {
      // Default to SOLO plan if none exists
      console.log("⚠️ No plan found, defaulting to SOLO (0 staff)");
      currentPlan = {
        name: "SOLO",
        staffLimit: 0
      };
    }
  } catch (error) {
    console.error("❌ Error loading shop plan:", error);
    return null;
  }
}

// ======================================================
// VALIDATE BEFORE ADDING STAFF - MAIN LOGIC
// ======================================================
async function validateBeforeAddingStaff() {
  try {
    if (!currentShopId) {
      console.log("❌ No shop ID found");
      showUpgradePrompt("Please login first");
      return false;
    }
    
    // Get shop plan if not loaded
    if (!currentPlan) {
      await loadShopPlan();
    }
    
    const planName = currentPlan?.name || "SOLO";
    const staffLimit = currentPlan?.staffLimit || 0;
    
    console.log(`🔍 Validating for plan: ${planName}, Staff Limit: ${staffLimit}`);
    
    // If SOLO plan - IMMEDIATELY show upgrade
    if (planName === "SOLO" || staffLimit === 0) {
      console.log("🚫 SOLO plan detected - cannot add staff");
      showUpgradePrompt("SOLO plan doesn't allow staff members");
      return false;
    }
    
    // For other plans, check current staff count
    const staffRef = collection(db, "Shops", currentShopId, "staff");
    const staffSnapshot = await getDocs(staffRef);
    const currentStaffCount = staffSnapshot.size;
    
    console.log(`👥 Current staff count: ${currentStaffCount}/${staffLimit}`);
    
    // Check if limit reached
    if (currentStaffCount >= staffLimit) {
      console.log(`🚫 Limit reached for ${planName} plan`);
      showUpgradePrompt(`You've reached the limit of ${staffLimit} staff members on your ${planName} plan`);
      return false;
    }
    
    console.log(`✅ Can add staff. ${staffLimit - currentStaffCount} slot(s) available`);
    return true;
    
  } catch (error) {
    console.error("❌ Error checking plan:", error);
    showUpgradePrompt("Error checking your plan. Please try again.");
    return false;
  }
}

// ======================================================
// SHOW UPGRADE PROMPT (Simple Version)
// ======================================================
function showUpgradePrompt(message) {
  console.log("📢 Upgrade Prompt:", message);
  
  // Create simple alert for now (we'll make it fancy later)
  const alertDiv = document.createElement('div');
  alertDiv.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: linear-gradient(135deg, #ff6b6b, #ff8e53);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    z-index: 10000;
    max-width: 350px;
    font-size: 14px;
    animation: slideInRight 0.3s ease;
  `;
  
  alertDiv.innerHTML = `
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
      <div style="font-size: 20px;">🚫</div>
      <div style="font-weight: 600;">Staff Limit</div>
    </div>
    <div style="margin-bottom: 15px;">${message}</div>
    <div style="display: flex; gap: 10px;">
      <button id="close-upgrade-alert" style="
        flex: 1;
        padding: 8px;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        border-radius: 6px;
        cursor: pointer;
      ">Close</button>
      <button id="view-upgrade-options" style="
        flex: 2;
        padding: 8px;
        background: white;
        border: none;
        color: #ff6b6b;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
      ">View Upgrade Options</button>
    </div>
  `;
  
  document.body.appendChild(alertDiv);
  
  // Add animation style
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideInRight {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `;
  document.head.appendChild(style);
  
  // Event listeners
  document.getElementById('close-upgrade-alert').onclick = () => {
    alertDiv.style.animation = 'slideInRight 0.3s ease reverse';
    setTimeout(() => {
      if (document.body.contains(alertDiv)) {
        document.body.removeChild(alertDiv);
      }
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    }, 300);
  };
  
  document.getElementById('view-upgrade-options').onclick = () => {
    // Remove alert
    if (document.body.contains(alertDiv)) {
      document.body.removeChild(alertDiv);
    }
    if (document.head.contains(style)) {
      document.head.removeChild(style);
    }
    
    // Show full upgrade options
    showFullUpgradeOptions();
  };
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (document.body.contains(alertDiv)) {
      alertDiv.style.animation = 'slideInRight 0.3s ease reverse';
      setTimeout(() => {
        if (document.body.contains(alertDiv)) {
          document.body.removeChild(alertDiv);
        }
        if (document.head.contains(style)) {
          document.head.removeChild(style);
        }
      }, 300);
    }
  }, 10000);
}

// ======================================================
// SHOW FULL UPGRADE OPTIONS
// ======================================================
function showFullUpgradeOptions() {
  console.log("🔄 Showing full upgrade options");
  
  const overlay = document.createElement('div');
  overlay.id = 'upgrade-options-overlay';
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.7);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    box-sizing: border-box;
    backdrop-filter: blur(5px);
  `;
  
  const currentPlanName = currentPlan?.name || "SOLO";
  const currentPlanData = PLANS[currentPlanName] || PLANS.SOLO;
  const nextPlan = PLANS[currentPlanData.nextUpgrade] || PLANS.BASIC;
  
  overlay.innerHTML = `
    <div style="
      background: white;
      border-radius: 16px;
      width: 100%;
      max-width: 500px;
      max-height: 90vh;
      overflow-y: auto;
      box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    ">
      <!-- Header -->
      <div style="
        padding: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 16px 16px 0 0;
      ">
        <h2 style="margin: 0; font-size: 20px; display: flex; align-items: center; gap: 10px;">
          <span>🚀</span>
          <span>Upgrade Your Plan</span>
        </h2>
        <p style="margin: 8px 0 0; opacity: 0.9; font-size: 14px;">
          Add more staff members to your team
        </p>
      </div>
      
      <!-- Current Plan -->
      <div style="padding: 20px; border-bottom: 1px solid #eee;">
        <div style="
          background: #f8f9fa;
          border-radius: 10px;
          padding: 15px;
          margin-bottom: 10px;
        ">
          <div style="font-weight: 600; color: #333; margin-bottom: 5px;">
            Current: ${currentPlanName} Plan
          </div>
          <div style="font-size: 13px; color: #666;">
            ${currentPlanData.description}
          </div>
          <div style="margin-top: 8px; font-size: 12px; color: #ff6b6b;">
            ✗ Cannot add staff members
          </div>
        </div>
      </div>
      
      <!-- Recommended Upgrade -->
      <div style="padding: 20px;">
        <h3 style="margin: 0 0 15px; font-size: 16px; color: #333;">
          Recommended Upgrade
        </h3>
        
        <div style="
          border: 2px solid #667eea;
          border-radius: 12px;
          padding: 20px;
          background: #f8faff;
          margin-bottom: 20px;
        ">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
              <div style="font-weight: 700; font-size: 18px; color: #667eea;">${nextPlan.name}</div>
              <div style="font-size: 13px; color: #666;">${nextPlan.description}</div>
            </div>
            <div style="
              background: #667eea;
              color: white;
              padding: 8px 16px;
              border-radius: 20px;
              font-weight: 700;
              font-size: 16px;
            ">
              KSh ${nextPlan.priceKES}/mo
            </div>
          </div>
          
          <div style="margin: 15px 0;">
            <div style="font-size: 14px; font-weight: 600; color: #333; margin-bottom: 8px;">Includes:</div>
            <div style="font-size: 13px; color: #555;">
              ✓ Up to ${nextPlan.staffLimit} staff members<br>
              ✓ All current features<br>
              ✓ Priority support
            </div>
          </div>
          
          <div style="
            background: #e8f5e9;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            text-align: center;
            font-size: 13px;
            color: #2e7d32;
            font-weight: 600;
          ">
            Get ${nextPlan.staffLimit} staff seats
          </div>
        </div>
        
        <!-- All Plans -->
        <div style="margin-top: 20px;">
          <div style="font-size: 14px; font-weight: 600; color: #333; margin-bottom: 10px;">All Available Plans:</div>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            ${Object.values(PLANS).map(plan => `
              <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 15px;
                background: ${plan.name === currentPlanName ? '#e8f5e9' : '#f8f9fa'};
                border-radius: 8px;
                border: 1px solid ${plan.name === currentPlanName ? '#4CAF50' : '#e0e0e0'};
              ">
                <div>
                  <div style="font-weight: 600; color: #333; font-size: 13px;">${plan.name}</div>
                  <div style="font-size: 12px; color: #666;">${plan.description}</div>
                </div>
                <div style="text-align: right;">
                  <div style="font-weight: 700; color: #333; font-size: 14px;">KSh ${plan.priceKES}/mo</div>
                  <div style="font-size: 11px; color: #666;">${plan.staffLimit} staff</div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
      
      <!-- Action Buttons -->
      <div style="
        padding: 20px;
        background: #f8f9fa;
        border-top: 1px solid #eee;
        display: flex;
        gap: 10px;
      ">
        <button id="close-upgrade" style="
          flex: 1;
          padding: 12px;
          border: 1px solid #ddd;
          background: white;
          color: #666;
          border-radius: 8px;
          font-weight: 600;
          font-size: 14px;
          cursor: pointer;
        ">Cancel</button>
        
        <button id="request-upgrade" style="
          flex: 2;
          padding: 12px;
          border: none;
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: white;
          border-radius: 8px;
          font-weight: 700;
          font-size: 14px;
          cursor: pointer;
        ">
          Request ${nextPlan.name} Upgrade
        </button>
      </div>
    </div>
  `;
  
  document.body.appendChild(overlay);
  
  // Event listeners
  document.getElementById('close-upgrade').onclick = () => {
    document.body.removeChild(overlay);
  };
  
  document.getElementById('request-upgrade').onclick = () => {
    requestPlanUpgrade(nextPlan.id);
    document.body.removeChild(overlay);
  };
  
  // Close on background click
  overlay.onclick = (e) => {
    if (e.target === overlay) {
      document.body.removeChild(overlay);
    }
  };
}

// ======================================================
// REQUEST PLAN UPGRADE
// ======================================================
async function requestPlanUpgrade(requestedPlanId) {
  console.log(`📝 Requesting upgrade to: ${requestedPlanId}`);
  
  // For now, just show a confirmation
  const requestedPlan = PLANS[requestedPlanId] || PLANS.BASIC;
  
  alert(`✅ Upgrade to ${requestedPlan.name} requested!\n\nOur team will contact you to complete the upgrade process.\n\nPrice: KSh ${requestedPlan.priceKES}/month\nStaff Limit: ${requestedPlan.staffLimit} members`);
  
  // In future, we'll save to Firestore:
  // await saveUpgradeRequest(requestedPlanId);
}

// ======================================================
// EXPORT FUNCTIONS
// ======================================================
window.validateBeforeAddingStaff = validateBeforeAddingStaff;

// Export for use in other modules
export { validateBeforeAddingStaff };