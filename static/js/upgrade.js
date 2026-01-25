// upgrade.js - Upgrade Management System with M-Pesa Payment
// Shows upgrade modal when staff limit is reached and collects M-Pesa payments

import { db } from "./firebase-config.js";
import { 
  doc, 
  getDoc, 
  setDoc,
  updateDoc,
  deleteDoc,
  query,
  where,
  getDocs,
  collection,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

console.log("✅ upgrade.js loaded");

let currentShopId = null;

// ======================================================
// PLAN DEFINITIONS
// ======================================================
const PLANS = {
  SOLO: {
    id: "SOLO",
    name: "Solo",
    priceKES: 0,
    staffLimit: 0,
    features: ["Owner access only", "Basic selling", "Stock management"],
    color: "#6B7280",
    icon: "👤",
    description: "Perfect for individual business owners"
  },
  BASIC: {
    id: "BASIC",
    name: "Basic",
    priceKES: 250,
    staffLimit: 5,
    features: ["Up to 5 staff", "All Solo features", "Staff management", "Basic reports"],
    color: "#3B82F6",
    icon: "👥",
    description: "Great for small teams",
    popular: true
  },
  TEAM: {
    id: "TEAM", 
    name: "Team",
    priceKES: 500,
    staffLimit: 10,
    features: ["Up to 10 staff", "All Basic features", "Advanced analytics", "Inventory forecasting"],
    color: "#8B5CF6",
    icon: "🏢",
    description: "For growing businesses"
  },
  BUSINESS: {
    id: "BUSINESS",
    name: "Business",
    priceKES: 2500,
    staffLimit: 20,
    features: ["Up to 20 staff", "All Team features", "Priority support", "Custom branding", "Export capabilities"],
    color: "#10B981",
    icon: "🏭",
    description: "For established businesses"
  },
  ENTERPRISE: {
    id: "ENTERPRISE",
    name: "Enterprise",
    priceKES: 5000,
    staffLimit: 100,
    features: ["Unlimited staff", "All Business features", "Dedicated support", "API access", "Custom integrations"],
    color: "#F59E0B",
    icon: "🚀",
    description: "For large organizations"
  }
};

// Upgrade paths
const UPGRADE_PATHS = {
  SOLO: ["BASIC", "TEAM", "BUSINESS", "ENTERPRISE"],
  BASIC: ["TEAM", "BUSINESS", "ENTERPRISE"],
  TEAM: ["BUSINESS", "ENTERPRISE"],
  BUSINESS: ["ENTERPRISE"],
  ENTERPRISE: []
};

// ======================================================
// CHECK FOR PENDING UPGRADE REQUESTS
// ======================================================
async function hasPendingUpgradeRequest() {
  try {
    if (!currentShopId) {
      const auth = getAuth();
      const user = auth.currentUser;
      if (user) {
        currentShopId = user.uid;
      } else {
        return false;
      }
    }
    
    const upgradeRequestsRef = collection(db, "Shops", currentShopId, "upgradeRequests");
    const q = query(
      upgradeRequestsRef,
      where("status", "in", ["pending_payment", "payment_submitted", "pending_verification"])
    );
    
    const querySnapshot = await getDocs(q);
    return !querySnapshot.empty;
  } catch (error) {
    console.error("❌ Error checking pending requests:", error);
    return false;
  }
}

// ======================================================
// GET PENDING UPGRADE REQUESTS
// ======================================================
async function getPendingUpgradeRequests() {
  try {
    if (!currentShopId) {
      const auth = getAuth();
      const user = auth.currentUser;
      if (user) {
        currentShopId = user.uid;
      } else {
        return [];
      }
    }
    
    const upgradeRequestsRef = collection(db, "Shops", currentShopId, "upgradeRequests");
    const q = query(
      upgradeRequestsRef,
      where("status", "in", ["pending_payment", "payment_submitted", "pending_verification"])
    );
    
    const querySnapshot = await getDocs(q);
    const requests = [];
    
    querySnapshot.forEach((doc) => {
      requests.push({
        id: doc.id,
        ...doc.data()
      });
    });
    
    return requests;
  } catch (error) {
    console.error("❌ Error getting pending requests:", error);
    return [];
  }
}

// ======================================================
// SHOW PENDING REQUESTS MODAL
// ======================================================
async function showPendingRequestsModal() {
  const pendingRequests = await getPendingUpgradeRequests();
  
  if (pendingRequests.length === 0) {
    return false; // No pending requests
  }
  
  // Create modal to show pending requests
  const modal = document.createElement('div');
  modal.id = 'pending-requests-modal';
  modal.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.85);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    box-sizing: border-box;
    backdrop-filter: blur(8px);
    animation: fadeIn 0.3s ease;
  `;
  
  modal.innerHTML = `
    <div style="
      background: white;
      border-radius: 20px;
      width: 100%;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
      animation: slideUp 0.4s ease;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
      padding-bottom: 100px; /* Space for cart */
    ">
      <!-- Header -->
      <div style="
        padding: 24px 32px;
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        color: white;
        border-radius: 20px 20px 0 0;
        position: relative;
      ">
        <div style="position: relative;">
          <h2 style="margin: 0; font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 10px;">
            <span>⏳</span>
            <span>Pending Upgrade Request</span>
          </h2>
          <p style="margin: 8px 0 0; font-size: 16px; opacity: 0.95;">
            You have ${pendingRequests.length} pending upgrade request${pendingRequests.length > 1 ? 's' : ''}
          </p>
        </div>
        <button id="close-pending-modal-btn" style="
          position: absolute;
          top: 20px;
          right: 20px;
          background: rgba(255,255,255,0.2);
          border: none;
          color: white;
          width: 36px;
          height: 36px;
          border-radius: 10px;
          font-size: 20px;
          cursor: pointer;
          transition: all 0.3s;
        ">×</button>
      </div>
      
      <!-- Content -->
      <div style="padding: 32px;">
        <div style="
          background: #fef3c7;
          border: 1px solid #fbbf24;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 24px;
          display: flex;
          align-items: flex-start;
          gap: 12px;
        ">
          <span style="font-size: 24px;">⚠️</span>
          <div>
            <div style="font-weight: 600; color: #92400e; margin-bottom: 4px;">
              Important Notice
            </div>
            <div style="color: #92400e; font-size: 14px;">
              You cannot create a new upgrade request while you have pending requests. 
              Please cancel or complete your existing request first.
            </div>
          </div>
        </div>
        
        <h3 style="margin: 0 0 20px; font-size: 18px; font-weight: 600; color: #333;">
          Your Pending Request${pendingRequests.length > 1 ? 's' : ''}
        </h3>
        
        ${pendingRequests.map(request => {
          const plan = PLANS[request.requestedPlan] || PLANS.SOLO;
          const statusMap = {
            'pending_payment': { text: 'Awaiting Payment', color: '#f59e0b', icon: '⏳' },
            'payment_submitted': { text: 'Payment Submitted', color: '#3b82f6', icon: '📤' },
            'pending_verification': { text: 'Verifying Payment', color: '#8b5cf6', icon: '🔍' }
          };
          const statusInfo = statusMap[request.status] || { text: 'Unknown', color: '#6b7280', icon: '❓' };
          
          return `
            <div style="
              border: 2px solid ${plan.color}40;
              border-radius: 16px;
              padding: 20px;
              margin-bottom: 16px;
              background: white;
            ">
              <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
                <div>
                  <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="
                      width: 36px;
                      height: 36px;
                      background: ${plan.color}20;
                      color: ${plan.color};
                      border-radius: 8px;
                      display: flex;
                      align-items: center;
                      justify-content: center;
                      font-size: 18px;
                    ">
                      ${plan.icon}
                    </div>
                    <div>
                      <div style="font-size: 20px; font-weight: 700; color: #333;">${plan.name}</div>
                      <div style="font-size: 14px; color: #666;">KSh ${plan.priceKES}/month</div>
                    </div>
                  </div>
                  <div style="font-size: 13px; color: #666;">
                    ${plan.staffLimit} staff members • Requested on ${new Date(request.requestedAt?.toDate()).toLocaleDateString()}
                  </div>
                </div>
                
                <div style="
                  background: ${statusInfo.color}15;
                  color: ${statusInfo.color};
                  padding: 6px 12px;
                  border-radius: 8px;
                  font-size: 12px;
                  font-weight: 600;
                  display: flex;
                  align-items: center;
                  gap: 6px;
                ">
                  <span>${statusInfo.icon}</span>
                  <span>${statusInfo.text}</span>
                </div>
              </div>
              
              <div style="display: flex; gap: 12px; margin-top: 16px;">
                <button onclick="completePayment('${request.id}')" style="
                  flex: 1;
                  padding: 10px;
                  background: ${plan.color};
                  border: none;
                  color: white;
                  border-radius: 8px;
                  font-size: 14px;
                  font-weight: 600;
                  cursor: pointer;
                  transition: all 0.3s;
                ">
                  ${request.status === 'pending_payment' ? 'Complete Payment' : 'View Details'}
                </button>
                
                <button onclick="cancelUpgradeRequest('${request.id}', '${plan.name}')" style="
                  padding: 10px 20px;
                  background: #f9fafb;
                  border: 2px solid #ef4444;
                  color: #ef4444;
                  border-radius: 8px;
                  font-size: 14px;
                  font-weight: 600;
                  cursor: pointer;
                  transition: all 0.3s;
                ">
                  Cancel
                </button>
              </div>
              
              ${request.mpesaReference ? `
                <div style="
                  margin-top: 16px;
                  padding: 12px;
                  background: #f0fdf4;
                  border: 1px solid #bbf7d0;
                  border-radius: 8px;
                  font-size: 13px;
                  color: #065f46;
                ">
                  <div style="display: flex; align-items: center; gap: 8px;">
                    <span>📋</span>
                    <span>M-Pesa Reference: <strong style="font-family: monospace;">${request.mpesaReference}</strong></span>
                  </div>
                </div>
              ` : ''}
            </div>
          `;
        }).join('')}
        
        <div style="
          background: #f8f9fa;
          border-radius: 12px;
          padding: 20px;
          margin-top: 24px;
          text-align: center;
        ">
          <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
            Need help with your upgrade request?
          </div>
          <div style="font-size: 13px; color: #999;">
            Contact support at support@yourbusiness.com or call +254 700 000 000
          </div>
        </div>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  addCartCompatibilityStyles();
  
  // Event listeners
  document.getElementById('close-pending-modal-btn').onclick = () => {
    closePendingRequestsModal();
  };
  
  modal.onclick = (e) => {
    if (e.target === modal) {
      closePendingRequestsModal();
    }
  };
  
  return true; // Modal was shown
}

// ======================================================
// CANCEL UPGRADE REQUEST
// ======================================================
async function cancelUpgradeRequest(requestId, planName) {
  if (!confirm(`Are you sure you want to cancel your upgrade request to ${planName}? This action cannot be undone.`)) {
    return;
  }
  
  try {
    if (!currentShopId) {
      const auth = getAuth();
      const user = auth.currentUser;
      if (user) {
        currentShopId = user.uid;
      } else {
        showToast('Please login to cancel request', 'error');
        return;
      }
    }
    
    const requestRef = doc(db, "Shops", currentShopId, "upgradeRequests", requestId);
    
    // Update status instead of deleting to keep record
    await updateDoc(requestRef, {
      status: "cancelled",
      cancelledAt: serverTimestamp(),
      cancelledBy: currentShopId,
      notes: "Cancelled by user before completing payment"
    });
    
    console.log("✅ Upgrade request cancelled:", requestId);
    showToast('Upgrade request cancelled successfully', 'success');
    
    // Close modal and refresh
    closePendingRequestsModal();
    
  } catch (error) {
    console.error("❌ Error cancelling upgrade request:", error);
    showToast('Failed to cancel request. Please try again.', 'error');
  }
}

// ======================================================
// COMPLETE PAYMENT FOR EXISTING REQUEST
// ======================================================
async function completePayment(requestId) {
  try {
    if (!currentShopId) {
      const auth = getAuth();
      const user = auth.currentUser;
      if (user) {
        currentShopId = user.uid;
      } else {
        showToast('Please login to continue', 'error');
        return;
      }
    }
    
    const requestRef = doc(db, "Shops", currentShopId, "upgradeRequests", requestId);
    const requestSnap = await getDoc(requestRef);
    
    if (!requestSnap.exists()) {
      showToast('Upgrade request not found', 'error');
      return;
    }
    
    const requestData = requestSnap.data();
    const plan = PLANS[requestData.requestedPlan] || PLANS.SOLO;
    
    // Show payment completion modal
    closePendingRequestsModal();
    showUpgradeConfirmation(plan, requestId);
    
  } catch (error) {
    console.error("❌ Error completing payment:", error);
    showToast('Failed to load request details', 'error');
  }
}

// ======================================================
// CLOSE PENDING REQUESTS MODAL
// ======================================================
function closePendingRequestsModal() {
  const modal = document.getElementById('pending-requests-modal');
  if (modal) {
    modal.style.animation = 'fadeIn 0.3s ease reverse';
    setTimeout(() => {
      if (document.body.contains(modal)) {
        document.body.removeChild(modal);
      }
    }, 300);
  }
}

// ======================================================
// SHOW UPGRADE MODAL (Main Function - Called from staffManager.js)
// ======================================================
async function showUpgradeModal(currentPlanName = "SOLO", reason = "Upgrade required") {
  console.log(`🔼 Showing upgrade modal for plan: ${currentPlanName}, reason: ${reason}`);
  
  // Check for pending requests first
  const hasPending = await hasPendingUpgradeRequest();
  if (hasPending) {
    const modalShown = await showPendingRequestsModal();
    if (modalShown) {
      return; // Don't show new upgrade modal if pending requests exist
    }
  }
  
  // Remove existing modal if present
  closeUpgradeModal();
  
  // Get current shop ID
  const auth = getAuth();
  const user = auth.currentUser;
  if (user) {
    currentShopId = user.uid;
  }
  
  // Get available upgrades based on current plan
  const currentPlanKey = Object.keys(PLANS).find(key => 
    PLANS[key].name.toUpperCase() === currentPlanName.toUpperCase()
  ) || "SOLO";
  
  const availableUpgrades = UPGRADE_PATHS[currentPlanKey] || ["BASIC", "TEAM", "BUSINESS", "ENTERPRISE"];
  
  if (availableUpgrades.length === 0) {
    showToast("You're already on the highest plan!", "info");
    return;
  }
  
  const upgradeOptions = availableUpgrades.map(planId => PLANS[planId]);
  const currentPlanData = PLANS[currentPlanKey] || PLANS.SOLO;
  
  // Create modal overlay
  const modal = document.createElement('div');
  modal.id = 'upgrade-modal-overlay';
  modal.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.85);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    padding-bottom: 120px; /* Extra space for cart */
    box-sizing: border-box;
    backdrop-filter: blur(8px);
    animation: fadeIn 0.3s ease;
    overflow-y: auto;
  `;
  
  modal.innerHTML = `
    <div style="
      background: white;
      border-radius: 20px;
      width: 100%;
      max-width: 800px;
      max-height: calc(90vh - 120px); /* Account for cart space */
      overflow-y: auto;
      animation: slideUp 0.4s ease;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
      margin: auto 0;
    ">
      <!-- Header -->
      <div style="
        padding: 24px 32px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 20px 20px 0 0;
        position: relative;
      ">
        <div style="position: relative;">
          <h2 style="margin: 0; font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 10px;">
            <span>🚀</span>
            <span>Upgrade Required</span>
          </h2>
          <p style="margin: 8px 0 0; font-size: 16px; opacity: 0.95;">
            ${reason === "Add staff members" ? "Add more team members to your business" : reason}
          </p>
        </div>
        <button id="close-upgrade-modal-btn" style="
          position: absolute;
          top: 20px;
          right: 20px;
          background: rgba(255,255,255,0.2);
          border: none;
          color: white;
          width: 36px;
          height: 36px;
          border-radius: 10px;
          font-size: 20px;
          cursor: pointer;
          transition: all 0.3s;
        ">×</button>
      </div>
      
      <!-- Current Plan -->
      <div style="padding: 24px 32px; border-bottom: 1px solid #e9ecef;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
          <div style="
            width: 48px;
            height: 48px;
            background: ${currentPlanData.color}20;
            color: ${currentPlanData.color};
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
          ">
            ${currentPlanData.icon}
          </div>
          <div>
            <div style="font-size: 14px; color: #666;">CURRENT PLAN</div>
            <div style="font-size: 20px; font-weight: 700; color: #333;">${currentPlanData.name}</div>
          </div>
        </div>
        <div style="color: #ef4444; font-weight: 600; display: flex; align-items: center; gap: 8px;">
          <span>⚠️</span>
          <span>Staff limit reached (${currentPlanData.staffLimit} staff maximum)</span>
        </div>
      </div>
      
      <!-- Upgrade Options -->
      <div style="padding: 32px;">
        <h3 style="margin: 0 0 24px; font-size: 20px; font-weight: 600; color: #333; text-align: center;">
          Choose Your Upgrade
        </h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
          ${upgradeOptions.map(plan => `
            <div class="upgrade-plan-card" data-plan-id="${plan.id}" style="
              border: 2px solid ${plan.popular ? plan.color : '#e5e7eb'};
              border-radius: 16px;
              padding: 24px;
              background: white;
              transition: all 0.3s ease;
              cursor: pointer;
              position: relative;
            ">
              ${plan.popular ? `
                <div style="
                  position: absolute;
                  top: -10px;
                  left: 50%;
                  transform: translateX(-50%);
                  background: ${plan.color};
                  color: white;
                  padding: 4px 12px;
                  border-radius: 12px;
                  font-size: 11px;
                  font-weight: 700;
                ">POPULAR</div>
              ` : ''}
              
              <!-- Plan Header -->
              <div style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                  <div style="
                    width: 40px;
                    height: 40px;
                    background: ${plan.color}20;
                    color: ${plan.color};
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                  ">
                    ${plan.icon}
                  </div>
                  <div style="font-size: 20px; font-weight: 700; color: #333;">${plan.name}</div>
                </div>
                <div style="font-size: 13px; color: #666;">${plan.description}</div>
              </div>
              
              <!-- Price -->
              <div style="margin-bottom: 16px;">
                <div style="font-size: 32px; font-weight: 800; color: ${plan.color}; line-height: 1;">
                  KSh ${plan.priceKES}
                </div>
                <div style="font-size: 13px; color: #666;">per month</div>
              </div>
              
              <!-- Staff Limit -->
              <div style="
                background: ${plan.color}10;
                border-radius: 10px;
                padding: 12px;
                margin-bottom: 16px;
                text-align: center;
              ">
                <div style="font-size: 13px; color: #666;">Staff Capacity</div>
                <div style="font-size: 18px; font-weight: 700; color: ${plan.color};">${plan.staffLimit} team members</div>
              </div>
              
              <!-- Features -->
              <div style="margin-bottom: 20px;">
                <div style="font-size: 13px; font-weight: 600; color: #333; margin-bottom: 8px;">Includes:</div>
                <ul style="margin: 0; padding-left: 0; list-style: none;">
                  ${plan.features.slice(0, 3).map(feature => `
                    <li style="
                      padding: 6px 0;
                      display: flex;
                      align-items: flex-start;
                      gap: 8px;
                      font-size: 13px;
                      color: #4b5563;
                    ">
                      <span style="color: ${plan.color}; font-size: 14px;">✓</span>
                      <span>${feature}</span>
                    </li>
                  `).join('')}
                </ul>
              </div>
              
              <!-- Select Button -->
              <button class="select-upgrade-btn" style="
                width: 100%;
                padding: 12px;
                background: ${plan.popular ? plan.color : '#f9fafb'};
                color: ${plan.popular ? 'white' : plan.color};
                border: 2px solid ${plan.color};
                border-radius: 10px;
                font-size: 14px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s;
              ">
                Select ${plan.name}
              </button>
            </div>
          `).join('')}
        </div>
        
        <!-- Contact Info -->
        <div style="
          margin-top: 32px;
          padding: 20px;
          background: #f8f9fa;
          border-radius: 12px;
          text-align: center;
        ">
          <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Need help deciding?</div>
          <div style="font-size: 13px; color: #999;">Contact us at support@yourbusiness.com or call +254 700 000 000</div>
        </div>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  // Add CSS animations and cart compatibility
  addModalStyles();
  addCartCompatibilityStyles();
  
  // ======================================================
  // EVENT LISTENERS
  // ======================================================
  
  // Close button
  document.getElementById('close-upgrade-modal-btn').onclick = () => {
    closeUpgradeModal();
  };
  
  // Close on background click
  modal.onclick = (e) => {
    if (e.target === modal) {
      closeUpgradeModal();
    }
  };
  
  // Close on Escape key
  document.addEventListener('keydown', function closeOnEscape(e) {
    if (e.key === 'Escape') {
      closeUpgradeModal();
      document.removeEventListener('keydown', closeOnEscape);
    }
  });
  
  // Plan card hover effects
  modal.querySelectorAll('.upgrade-plan-card').forEach(card => {
    // Hover effect
    card.onmouseenter = () => {
      card.style.transform = 'translateY(-5px)';
      card.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)';
    };
    
    card.onmouseleave = () => {
      card.style.transform = 'translateY(0)';
      card.style.boxShadow = 'none';
    };
    
    // Click on card
    card.addEventListener('click', (e) => {
      if (!e.target.classList.contains('select-upgrade-btn')) {
        const selectBtn = card.querySelector('.select-upgrade-btn');
        selectBtn.style.background = card.dataset.planId === 'BASIC' ? '#3B82F6' : '#10B981';
        selectBtn.style.color = 'white';
        setTimeout(() => {
          selectBtn.click();
        }, 100);
      }
    });
    
    // Select button
    const selectBtn = card.querySelector('.select-upgrade-btn');
    selectBtn.onclick = async (e) => {
      e.stopPropagation();
      const planId = card.dataset.planId;
      
      // Final check for pending requests
      const hasPending = await hasPendingUpgradeRequest();
      if (hasPending) {
        showToast('You have a pending upgrade request. Please cancel it first.', 'error');
        return;
      }
      
      initiateUpgrade(planId);
    };
  });
}

// ======================================================
// ADD MODAL STYLES
// ======================================================
function addModalStyles() {
  const styleId = 'upgrade-modal-styles';
  if (document.getElementById(styleId)) return;
  
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    @keyframes slideUp {
      from { transform: translateY(20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes popIn {
      0% { transform: scale(0.9); opacity: 0; }
      100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes slideIn {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
      from { transform: translateX(0); opacity: 1; }
      to { transform: translateX(100%); opacity: 0; }
    }
    
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); }
    }
  `;
  
  document.head.appendChild(style);
}

// ======================================================
// ADD CART COMPATIBILITY STYLES
// ======================================================
function addCartCompatibilityStyles() {
  const styleId = 'cart-compatibility-styles';
  if (document.getElementById(styleId)) return;
  
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    /* Cart compatibility styles - prevent overlay conflicts */
    #upgrade-modal-overlay,
    #upgrade-confirmation,
    #pending-requests-modal {
      z-index: 2147483648 !important; /* Higher than cart z-index */
      padding-bottom: 120px !important; /* Space for cart */
      align-items: flex-start !important;
      overflow-y: auto !important;
    }
    
    #upgrade-modal-overlay > div,
    #upgrade-confirmation > div,
    #pending-requests-modal > div {
      margin-top: 20px !important;
      margin-bottom: 20px !important;
      max-height: calc(100vh - 140px) !important;
      overflow-y: auto !important;
    }
    
    /* Mobile optimization */
    @media (max-height: 700px) {
      #upgrade-modal-overlay,
      #upgrade-confirmation,
      #pending-requests-modal {
        align-items: flex-start !important;
        padding-top: 60px !important;
      }
      
      #upgrade-modal-overlay > div,
      #upgrade-confirmation > div,
      #pending-requests-modal > div {
        max-height: calc(100vh - 180px) !important;
      }
    }
    
    /* Small screens */
    @media (max-width: 640px) {
      #upgrade-modal-overlay,
      #upgrade-confirmation,
      #pending-requests-modal {
        padding: 10px !important;
        padding-bottom: 100px !important;
      }
      
      #upgrade-modal-overlay > div,
      #upgrade-confirmation > div,
      #pending-requests-modal > div {
        max-height: calc(100vh - 120px) !important;
      }
    }
  `;
  
  document.head.appendChild(style);
}

// ======================================================
// CLOSE UPGRADE MODAL
// ======================================================
function closeUpgradeModal() {
  const modal = document.getElementById('upgrade-modal-overlay');
  if (modal) {
    modal.style.animation = 'fadeIn 0.3s ease reverse';
    setTimeout(() => {
      if (document.body.contains(modal)) {
        document.body.removeChild(modal);
      }
    }, 300);
  }
  
  // Remove styles
  const style = document.getElementById('upgrade-modal-styles');
  if (style) {
    document.head.removeChild(style);
  }
}

// ======================================================
// INITIATE UPGRADE PROCESS
// ======================================================
async function initiateUpgrade(planId) {
  try {
    if (!currentShopId) {
      const auth = getAuth();
      const user = auth.currentUser;
      if (user) {
        currentShopId = user.uid;
      } else {
        showToast('Please login to upgrade', 'error');
        return;
      }
    }
    
    // Final check for pending requests
    const hasPending = await hasPendingUpgradeRequest();
    if (hasPending) {
      showToast('You have a pending upgrade request. Please cancel it first.', 'error');
      return;
    }
    
    const requestedPlan = PLANS[planId];
    if (!requestedPlan) {
      showToast('Invalid plan selected', 'error');
      return;
    }
    
    console.log(`📝 Requesting upgrade to: ${requestedPlan.name}`);
    
    // Save upgrade request to Firestore
    const upgradeRequestId = await saveUpgradeRequest(planId);
    
    // Show confirmation with M-Pesa payment instructions
    showUpgradeConfirmation(requestedPlan, upgradeRequestId);
    
    // Close the modal
    closeUpgradeModal();
    
  } catch (error) {
    console.error("❌ Error initiating upgrade:", error);
    showToast('Failed to process upgrade request. Please try again.', 'error');
  }
}

// ======================================================
// SAVE UPGRADE REQUEST TO FIRESTORE
// ======================================================
async function saveUpgradeRequest(planId) {
  try {
    const requestedPlan = PLANS[planId];
    
    // Create a unique request ID
    const requestId = `upgrade_${currentShopId}_${Date.now()}`;
    
    const upgradeRef = doc(db, "Shops", currentShopId, "upgradeRequests", requestId);
    
    const upgradeData = {
      shopId: currentShopId,
      requestedPlan: planId,
      planName: requestedPlan.name,
      priceKES: requestedPlan.priceKES,
      staffLimit: requestedPlan.staffLimit,
      status: "pending_payment",
      requestedAt: serverTimestamp(),
      paymentSubmittedAt: null,
      mpesaReference: null,
      paymentStatus: "pending",
      processedBy: null,
      processedAt: null,
      notes: "Requested via staff manager"
    };
    
    await setDoc(upgradeRef, upgradeData);
    
    console.log("✅ Upgrade request saved to Firestore:", requestId);
    return requestId;
    
  } catch (error) {
    console.error("❌ Error saving upgrade request:", error);
    throw error;
  }
}

// ======================================================
// SHOW UPGRADE CONFIRMATION WITH M-PESA PAYMENT
// ======================================================
function showUpgradeConfirmation(plan, upgradeRequestId) {
  const overlay = document.createElement('div');
  overlay.id = 'upgrade-confirmation';
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.9);
    z-index: 11000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    padding-bottom: 120px;
    backdrop-filter: blur(10px);
    overflow-y: auto;
  `;
  
  overlay.innerHTML = `
    <div style="
      background: white;
      border-radius: 20px;
      width: 100%;
      max-width: 500px;
      padding: 32px;
      text-align: center;
      animation: popIn 0.3s ease;
      margin: auto 0;
      position: relative;
    ">
      <!-- Close button in top right corner -->
      <button id="close-confirmation-btn" style="
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(0,0,0,0.1);
        border: none;
        color: #666;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.3s;
        z-index: 1;
      ">×</button>
      
      <div style="
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        margin: 0 auto 20px;
      ">
        ✓
      </div>
      
      <h2 style="margin: 0 0 12px; font-size: 24px; font-weight: 700; color: #333;">
        Upgrade Requested!
      </h2>
      
      <p style="color: #666; font-size: 15px; line-height: 1.6; margin-bottom: 24px;">
        Your request to upgrade to <strong style="color: ${plan.color};">${plan.name}</strong> has been received.
      </p>
      
      <div style="
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        text-align: left;
      ">
        <div style="font-size: 28px; font-weight: 800; color: #333; text-align: center; margin-bottom: 12px;">
          KSh ${plan.priceKES}<span style="font-size: 14px; color: #666;">/month</span>
        </div>
        
        <div style="font-size: 14px; color: #666; text-align: center; margin-bottom: 16px;">
          Up to ${plan.staffLimit} staff members
        </div>
        
        <!-- M-Pesa Payment Instructions -->
        <div style="
          background: white;
          border: 2px solid #10B981;
          border-radius: 10px;
          padding: 16px;
          margin-top: 16px;
        ">
          <div style="font-size: 15px; font-weight: 600; color: #333; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span>💳</span>
            <span>Pay via M-Pesa:</span>
          </div>
          
          <div style="font-size: 14px; color: #666; margin-bottom: 12px;">
            Send <strong style="color: #333;">KSh ${plan.priceKES}</strong> to:
          </div>
          
          <div style="
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            margin-bottom: 12px;
          ">
            <div style="font-size: 13px; color: #666;">M-Pesa Number</div>
            <div style="font-size: 22px; font-weight: 700; color: #10B981; font-family: monospace; letter-spacing: 1px;">
              0114932232
            </div>
          </div>
          
          <div style="font-size: 13px; color: #666; margin-bottom: 12px;">
            Then enter your M-Pesa reference code below:
          </div>
          
          <!-- M-Pesa Reference Input -->
          <div style="display: flex; gap: 8px;">
            <input type="text" 
                   id="mpesa-reference" 
                   placeholder="e.g., RCV2H88ABC or any reference"
                   maxlength="50"
                   style="
                     flex: 1;
                     padding: 12px;
                     border: 2px solid #e5e7eb;
                     border-radius: 8px;
                     font-size: 14px;
                     font-family: monospace;
                     letter-spacing: 1px;
                     text-transform: uppercase;
                   "
                   oninput="this.value = this.value.toUpperCase()">
            
            <button id="submit-reference-btn" style="
              padding: 12px 16px;
              background: ${plan.color};
              border: none;
              color: white;
              border-radius: 8px;
              font-size: 14px;
              font-weight: 600;
              cursor: pointer;
              white-space: nowrap;
            ">
              Submit
            </button>
          </div>
          
          <div id="reference-error" style="
            margin-top: 8px;
            color: #ef4444;
            font-size: 13px;
            display: none;
          ">
            Please enter your M-Pesa reference code
          </div>
        </div>
      </div>
      
      <div style="font-size: 13px; color: #666; margin-bottom: 24px; text-align: left;">
        <div style="display: flex; align-items: flex-start; gap: 8px; margin-bottom: 8px;">
          <span>📧</span>
          <span>You'll receive confirmation email once payment is verified</span>
        </div>
        <div style="display: flex; align-items: flex-start; gap: 8px;">
          <span>⏱️</span>
          <span>Processing takes 5-10 minutes after payment submission</span>
        </div>
      </div>
      
      <!-- Hidden upgrade request ID -->
      <input type="hidden" id="upgrade-request-id" value="${upgradeRequestId}">
    </div>
  `;
  
  document.body.appendChild(overlay);
  addCartCompatibilityStyles();
  
  // ======================================================
  // EVENT LISTENERS
  // ======================================================
  
  // Close button (top right X)
  document.getElementById('close-confirmation-btn').onclick = () => {
    if (document.body.contains(overlay)) {
      document.body.removeChild(overlay);
    }
  };
  
  // Submit reference button
  document.getElementById('submit-reference-btn').onclick = async () => {
    await submitMpesaReference(plan, upgradeRequestId);
  };
  
  // Allow Enter key to submit
  document.getElementById('mpesa-reference').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      document.getElementById('submit-reference-btn').click();
    }
  });
  
  // Focus on the input
  setTimeout(() => {
    document.getElementById('mpesa-reference').focus();
  }, 300);
}

// ======================================================
// SUBMIT M-PESA REFERENCE
// ======================================================
async function submitMpesaReference(plan, upgradeRequestId) {
  const mpesaRef = document.getElementById('mpesa-reference').value.trim().toUpperCase();
  const errorDiv = document.getElementById('reference-error');
  
  // SIMPLIFIED VALIDATION - Only check if it's not empty
  if (!mpesaRef || mpesaRef.length === 0) {
    errorDiv.textContent = 'Please enter your M-Pesa reference code';
    errorDiv.style.display = 'block';
    return;
  }
  
  // Show loading state
  const submitBtn = document.getElementById('submit-reference-btn');
  const originalText = submitBtn.textContent;
  submitBtn.textContent = 'Submitting...';
  submitBtn.disabled = true;
  errorDiv.style.display = 'none';
  
  try {
    // Create payment record in Firestore
    const paymentRef = doc(collection(db, "Shops", currentShopId, "payments"));
    
    const paymentData = {
      shopId: currentShopId,
      upgradeRequestId: upgradeRequestId,
      planId: plan.id,
      planName: plan.name,
      amount: plan.priceKES,
      mpesaReference: mpesaRef,
      mpesaNumber: "0114932232",
      status: "pending",
      paymentStatus: "submitted",
      submittedAt: serverTimestamp(),
      verifiedAt: null,
      verifiedBy: null,
      metadata: {
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        ipAddress: await getClientIP()
      }
    };
    
    await setDoc(paymentRef, paymentData);
    
    console.log("✅ M-Pesa reference saved to Firestore:", mpesaRef);
    
    // Update upgrade request status
    const upgradeRef = doc(db, "Shops", currentShopId, "upgradeRequests", upgradeRequestId);
    await updateDoc(upgradeRef, {
      status: "payment_submitted",
      paymentStatus: "pending_verification",
      mpesaReference: mpesaRef,
      paymentSubmittedAt: serverTimestamp(),
      updatedAt: serverTimestamp()
    });
    
    console.log("✅ Upgrade request updated with payment reference");
    
    // Show success message
    showPaymentSubmittedSuccess(plan, mpesaRef);
    
  } catch (error) {
    console.error("❌ Error saving M-Pesa reference:", error);
    errorDiv.textContent = 'Error saving reference. Please try again or contact support.';
    errorDiv.style.display = 'block';
    
    // Reset button
    submitBtn.textContent = originalText;
    submitBtn.disabled = false;
  }
}

// ======================================================
// SHOW PAYMENT SUBMITTED SUCCESS
// ======================================================
function showPaymentSubmittedSuccess(plan, mpesaRef) {
  const overlay = document.getElementById('upgrade-confirmation');
  if (!overlay) return;
  
  overlay.innerHTML = `
    <div style="
      background: white;
      border-radius: 20px;
      width: 100%;
      max-width: 500px;
      padding: 32px;
      text-align: center;
      animation: popIn 0.3s ease;
      position: relative;
    ">
      <!-- Close button in top right corner -->
      <button id="close-success-btn" style="
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(0,0,0,0.1);
        border: none;
        color: #666;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.3s;
        z-index: 1;
      ">×</button>
      
      <div style="
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        margin: 0 auto 20px;
        animation: pulse 2s infinite;
      ">
        ✓
      </div>
      
      <h2 style="margin: 0 0 12px; font-size: 24px; font-weight: 700; color: #333;">
        Payment Submitted!
      </h2>
      
      <p style="color: #666; font-size: 15px; line-height: 1.6; margin-bottom: 24px;">
        Your M-Pesa payment for <strong style="color: ${plan.color};">${plan.name}</strong> has been received.
      </p>
      
      <div style="
        background: #f0fdf4;
        border: 2px solid #bbf7d0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
      ">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
          <div style="text-align: left;">
            <div style="font-size: 13px; color: #666;">Reference Code</div>
            <div style="font-size: 18px; font-weight: 700; color: #333; font-family: monospace;">${mpesaRef}</div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 13px; color: #666;">Amount</div>
            <div style="font-size: 24px; font-weight: 800; color: #333;">KSh ${plan.priceKES}</div>
          </div>
        </div>
        
        <div style="
          background: white;
          border-radius: 8px;
          padding: 12px;
          font-size: 13px;
          color: #666;
        ">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span style="color: #10B981;">⏳</span>
            <span>Status: <strong style="color: #f59e0b;">Pending Verification</strong></span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #10B981;">📧</span>
            <span>Confirmation email will be sent shortly</span>
          </div>
        </div>
      </div>
      
      <div style="font-size: 13px; color: #666; margin-bottom: 24px; text-align: left;">
        <div style="font-weight: 600; color: #333; margin-bottom: 8px;">Next steps:</div>
        <ol style="margin: 0; padding-left: 20px; color: #666; font-size: 13px; line-height: 1.8;">
          <li>We verify your payment against M-Pesa records</li>
          <li>Your plan is upgraded automatically</li>
          <li>New staff limits take effect immediately</li>
          <li>You'll receive confirmation within 5-10 minutes</li>
        </ol>
      </div>
    </div>
  `;
  
  // Close button
  document.getElementById('close-success-btn').onclick = () => {
    if (document.body.contains(overlay)) {
      document.body.removeChild(overlay);
    }
  };
}

// ======================================================
// GET CLIENT IP (Optional - for logging)
// ======================================================
async function getClientIP() {
  try {
    const response = await fetch('https://api.ipify.org?format=json');
    const data = await response.json();
    return data.ip;
  } catch (error) {
    console.warn("Could not fetch IP address:", error);
    return "unknown";
  }
}

// ======================================================
// SHOW TOAST NOTIFICATION
// ======================================================
function showToast(message, type = 'success') {
  // Remove existing toast
  const existingToast = document.getElementById('upgrade-toast');
  if (existingToast) {
    document.body.removeChild(existingToast);
  }

  const toast = document.createElement('div');
  toast.id = 'upgrade-toast';
  toast.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#3B82F6'};
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    z-index: 2147483649; /* Higher than all modals */
    font-weight: 500;
    font-size: 14px;
    animation: slideIn 0.3s ease;
  `;
  toast.textContent = message;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
      if (document.body.contains(toast)) {
        document.body.removeChild(toast);
      }
    }, 300);
  }, 3000);
}

// ======================================================
// EXPORT FUNCTIONS
// ======================================================
window.showUpgradeModal = showUpgradeModal;
window.closeUpgradeModal = closeUpgradeModal;
window.initiateUpgrade = initiateUpgrade;
window.cancelUpgradeRequest = cancelUpgradeRequest;
window.completePayment = completePayment;
window.hasPendingUpgradeRequest = hasPendingUpgradeRequest;
window.showPendingRequestsModal = showPendingRequestsModal;
window.PLANS = PLANS;

export { 
  showUpgradeModal, 
  closeUpgradeModal, 
  initiateUpgrade, 
  cancelUpgradeRequest,
  completePayment,
  hasPendingUpgradeRequest,
  showPendingRequestsModal,
  PLANS 
};