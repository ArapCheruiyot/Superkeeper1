// access-controls.js - SIMPLE ACCESS CONTROL BASED ON LEVELS
const ACCESS_LEVELS = {
    1: { // Basic - Can only sell
        sell: true,
        manageStock: false,
        businessIntelligence: false,
        settings: false
    },
    2: { // Intermediate - Can sell & manage stock
        sell: true,
        manageStock: true,
        businessIntelligence: false,
        settings: false
    },
    3: { // Advanced - Can sell, manage stock & view reports
        sell: true,
        manageStock: true,
        businessIntelligence: true,
        settings: false
    },
    4: { // Admin - Full access (Owner/Manager)
        sell: true,
        manageStock: true,
        businessIntelligence: true,
        settings: true
    }
};

// Apply access controls based on level
function applyAccessControls(accessLevel) {
    const level = ACCESS_LEVELS[accessLevel] || ACCESS_LEVELS[1];
    
    console.log(`🔐 Applying access level ${accessLevel}:`, level);
    
    // Control Sell button
    const sellBtn = document.getElementById('sell-btn');
    if (sellBtn) {
        sellBtn.style.display = level.sell ? '' : 'none';
    }
    
    // Control Manage Stock button
    const stockBtn = document.getElementById('manage-stock-btn');
    if (stockBtn) {
        stockBtn.style.display = level.manageStock ? '' : 'none';
    }
    
    // Control Business Intelligence button
    const biBtn = document.getElementById('business-intelligence-btn');
    if (biBtn) {
        biBtn.style.display = level.businessIntelligence ? '' : 'none';
    }
    
    // Control Settings gear
    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
        settingsBtn.style.display = level.settings ? '' : 'none';
    }
}

// Check if user is staff (for future use)
function isUserStaff() {
    return localStorage.getItem('isStaff') === 'true';
}

// Get access level from localStorage
function getAccessLevel() {
    const level = parseInt(localStorage.getItem('accessLevel')) || 1;
    return Math.min(Math.max(level, 1), 4); // Ensure between 1-4
}

// Export functions
window.applyAccessControls = applyAccessControls;
window.isUserStaff = isUserStaff;
window.getAccessLevel = getAccessLevel;
window.ACCESS_LEVELS = ACCESS_LEVELS;