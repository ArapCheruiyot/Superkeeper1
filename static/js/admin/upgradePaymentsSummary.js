import { db } from "../firebase-config.js";
import { 
    collection, 
    getDocs, 
    onSnapshot,
    query,
    where
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

// Store the unsubscribe functions to clean up later
let unsubscribeFunctions = [];
let isListening = false;

async function loadUpgradeSummary() {
    const container = document.getElementById('upgrade-summary');
    container.innerHTML = '<div class="loading">Loading upgrade requests...</div>';

    try {
        console.log("📊 Loading upgrade requests from all shops...");

        // 1️⃣ Get all shops first to get shop names
        const shopsSnapshot = await getDocs(collection(db, "Shops"));
        if (shopsSnapshot.empty) {
            container.innerHTML = '<p>No shops found.</p>';
            updateStats(0, 0, 0, 0);
            return;
        }

        // Create a map of shopId → shopName
        const shopMap = {};
        shopsSnapshot.forEach(shopDoc => {
            const shopData = shopDoc.data();
            shopMap[shopDoc.id] = shopData.name || shopData.shopName || "Unknown Shop";
        });

        console.log("🏪 Shop map:", shopMap);

        // Clear any existing listeners
        cleanupListeners();

        // 2️⃣ Set up real-time listeners for each shop's upgradeRequests subcollection
        const allRequests = [];
        const shopPromises = [];

        for (const shopId in shopMap) {
            const promise = setupShopListener(shopId, shopMap[shopId], allRequests);
            shopPromises.push(promise);
        }

        // Wait for all listeners to be set up
        await Promise.all(shopPromises);

        // Process the initial data
        processAndDisplayData(allRequests, shopMap);

        // Start listening for changes
        isListening = true;
        console.log("✅ Real-time listeners activated");

    } catch (error) {
        console.error("❌ Error loading upgrade requests:", error);
        container.innerHTML = `
            <div style="color: red; padding: 1rem; background: #f8d7da; border-radius: 5px;">
                <h3>Failed to load upgrade requests</h3>
                <p>Error: ${error.message}</p>
                <button onclick="loadUpgradeSummary()" style="padding: 5px 10px; margin-top: 10px;">Retry</button>
            </div>
        `;
        updateStats(0, 0, 0, 0);
    }
}

function setupShopListener(shopId, shopName, allRequestsArray) {
    return new Promise((resolve) => {
        try {
            const upgradeRequestsRef = collection(db, `Shops/${shopId}/upgradeRequests`);
            
            // Set up real-time listener
            const unsubscribe = onSnapshot(upgradeRequestsRef, (snapshot) => {
                console.log(`🔄 Real-time update for shop ${shopId}:`, snapshot.size, "documents");
                
                // Remove existing requests from this shop
                const existingIndexes = [];
                allRequestsArray.forEach((req, index) => {
                    if (req.shopId === shopId) {
                        existingIndexes.push(index);
                    }
                });
                
                // Remove from end to beginning to maintain indexes
                existingIndexes.reverse().forEach(index => {
                    allRequestsArray.splice(index, 1);
                });
                
                // Add new/updated requests
                snapshot.forEach(reqDoc => {
                    const req = reqDoc.data();
                    
                    allRequestsArray.push({
                        id: reqDoc.id,
                        shopId,
                        shopName: req.shopName || shopName || "Unknown Shop",
                        requestedPlan: req.requestedPlan || req.planName || "N/A",
                        status: req.status || req.paymentStatus || "unknown",
                        mpesaReference: req.mpesaReference || "N/A",
                        requestedAt: req.requestedAt || req.timestamp || null,
                        verifiedAt: req.verifiedAt || null,
                        paymentSubmittedAt: req.paymentSubmittedAt || null,
                        priceKES: req.priceKES || "N/A",
                        staffLimit: req.staffLimit || "N/A",
                        _raw: req,
                        _updatedAt: new Date() // Track when this record was last updated
                    });
                });
                
                // Process and display updated data
                processAndDisplayData(allRequestsArray);
                
                // Show notification for new updates
                showUpdateNotification(`Updated: ${shopName}`);
            }, (error) => {
                console.error(`❌ Listener error for shop ${shopId}:`, error);
            });
            
            // Store unsubscribe function for cleanup
            unsubscribeFunctions.push(unsubscribe);
            resolve();
            
        } catch (error) {
            console.error(`⚠️ Error setting up listener for shop ${shopId}:`, error.message);
            resolve(); // Resolve anyway to continue with other shops
        }
    });
}

function processAndDisplayData(allRequests, shopMap = null) {
    console.log("🔄 Processing data:", allRequests.length, "requests");
    
    if (allRequests.length === 0) {
        document.getElementById('upgrade-summary').innerHTML = "<p>No upgrade requests found.</p>";
        updateStats(0, 0, 0, 0);
        return;
    }

    // 3️⃣ Categorize properly
    // Paid/Submitted statuses: 'submitted', 'payment_submitted', 'pending_verification'
    const paidStatuses = ['submitted', 'payment_submitted', 'pending_verification'];
    
    const paid = allRequests.filter(r => paidStatuses.includes(r.status));
    const requestedOnly = allRequests.filter(r => !paidStatuses.includes(r.status));
    const verified = allRequests.filter(r => r.verifiedAt !== null);

    console.log(`📊 Stats: Total=${allRequests.length}, Paid=${paid.length}, RequestedOnly=${requestedOnly.length}, Verified=${verified.length}`);

    // 4️⃣ Update statistics
    updateStats(allRequests.length, requestedOnly.length, paid.length, verified.length);

    // 5️⃣ Build HTML
    let html = '';

    // Paid / Waiting Verification
    if (paid.length > 0) {
        // Sort by most recent first
        const sortedPaid = [...paid].sort((a, b) => {
            const dateA = a.paymentSubmittedAt || a.requestedAt;
            const dateB = b.paymentSubmittedAt || b.requestedAt;
            return getTimestamp(dateB) - getTimestamp(dateA);
        });
        
        html += `
            <div style="margin-bottom: 2rem;">
                <h3>💰 Paid / Waiting Verification (${paid.length})</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Shop Name</th>
                            <th>Requested Plan</th>
                            <th>Status</th>
                            <th>MPESA Ref</th>
                            <th>Amount</th>
                            <th>Submitted At</th>
                            <th>Last Update</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sortedPaid.map(r => `
                            <tr>
                                <td><strong>${r.shopName}</strong></td>
                                <td>${r.requestedPlan}</td>
                                <td><span style="padding: 3px 8px; background: #d4edda; border-radius: 4px; color: #155724;">${r.status}</span></td>
                                <td><code>${r.mpesaReference}</code></td>
                                <td>KES ${r.priceKES}</td>
                                <td>${formatDate(r.paymentSubmittedAt || r.requestedAt)}</td>
                                <td><small style="color: #6c757d;">${formatTimeAgo(r._updatedAt)}</small></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    // Just Requested (not paid yet)
    if (requestedOnly.length > 0) {
        // Sort by most recent first
        const sortedRequested = [...requestedOnly].sort((a, b) => {
            return getTimestamp(b.requestedAt) - getTimestamp(a.requestedAt);
        });
        
        html += `
            <div style="margin-bottom: 2rem;">
                <h3>📝 Just Requested - Awaiting Payment (${requestedOnly.length})</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Shop Name</th>
                            <th>Requested Plan</th>
                            <th>Status</th>
                            <th>Requested At</th>
                            <th>Price</th>
                            <th>Last Update</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sortedRequested.map(r => `
                            <tr>
                                <td><strong>${r.shopName}</strong></td>
                                <td>${r.requestedPlan}</td>
                                <td><span style="padding: 3px 8px; background: #fff3cd; border-radius: 4px; color: #856404;">${r.status}</span></td>
                                <td>${formatDate(r.requestedAt)}</td>
                                <td>KES ${r.priceKES}</td>
                                <td><small style="color: #6c757d;">${formatTimeAgo(r._updatedAt)}</small></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    // Already Verified
    if (verified.length > 0) {
        // Sort by most recent verification first
        const sortedVerified = [...verified].sort((a, b) => {
            return getTimestamp(b.verifiedAt) - getTimestamp(a.verifiedAt);
        });
        
        html += `
            <div style="margin-bottom: 2rem;">
                <h3>✅ Already Verified (${verified.length})</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Shop Name</th>
                            <th>Plan</th>
                            <th>Verified At</th>
                            <th>MPESA Ref</th>
                            <th>Last Update</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sortedVerified.map(r => `
                            <tr>
                                <td><strong>${r.shopName}</strong></td>
                                <td>${r.requestedPlan}</td>
                                <td>${formatDate(r.verifiedAt)}</td>
                                <td><code>${r.mpesaReference}</code></td>
                                <td><small style="color: #6c757d;">${formatTimeAgo(r._updatedAt)}</small></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    // Status indicator
    const statusIndicator = isListening ? 
        '<span style="color: green;">● Live</span>' : 
        '<span style="color: orange;">● Manual</span>';
    
    html += `
        <div style="margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 5px; font-size: 0.9rem;">
            <h4>📋 Dashboard Status ${statusIndicator}</h4>
            <p>Found ${allRequests.length} upgrade request(s) from ${shopMap ? Object.keys(shopMap).length : 'multiple'} shop(s)</p>
            <p>Last updated: ${new Date().toLocaleTimeString()}</p>
            <button onclick="toggleAutoRefresh()" style="padding: 5px 10px; margin-top: 5px; font-size: 0.8rem;">
                ${isListening ? '⏸️ Pause Updates' : '▶️ Resume Updates'}
            </button>
        </div>
    `;

    document.getElementById('upgrade-summary').innerHTML = html;
}

function updateStats(total, requested, paidVerify, verified) {
    document.getElementById('total-requests').textContent = total;
    document.getElementById('awaiting-payment').textContent = requested;
    document.getElementById('paid-verify').textContent = paidVerify;
    document.getElementById('verified').textContent = verified;
}

function formatDate(timestamp) {
    if (!timestamp) return "N/A";
    
    try {
        let date;
        if (timestamp.toDate) {
            // Firestore timestamp
            date = timestamp.toDate();
        } else if (timestamp.seconds) {
            // Timestamp object
            date = new Date(timestamp.seconds * 1000);
        } else if (typeof timestamp === 'string') {
            // ISO string
            date = new Date(timestamp);
        } else {
            return "Invalid date";
        }
        
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    } catch (e) {
        console.error("Date formatting error:", e);
        return "Date error";
    }
}

function getTimestamp(dateObj) {
    if (!dateObj) return 0;
    
    try {
        if (dateObj.toDate) {
            return dateObj.toDate().getTime();
        } else if (dateObj.seconds) {
            return dateObj.seconds * 1000;
        } else if (typeof dateObj === 'string') {
            return new Date(dateObj).getTime();
        } else if (dateObj instanceof Date) {
            return dateObj.getTime();
        }
    } catch (e) {
        console.error("Timestamp error:", e);
    }
    return 0;
}

function formatTimeAgo(date) {
    if (!date) return "N/A";
    
    const now = new Date();
    const updateDate = date instanceof Date ? date : new Date(date);
    const diffMs = now - updateDate;
    const diffSec = Math.floor(diffMs / 1000);
    
    if (diffSec < 60) return "just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)} min ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} hr ago`;
    return `${Math.floor(diffSec / 86400)} days ago`;
}

function showUpdateNotification(message) {
    // Create or update notification element
    let notification = document.getElementById('update-notification');
    
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'update-notification';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        document.body.appendChild(notification);
        
        // Add CSS animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
    
    notification.textContent = `🔄 ${message}`;
    notification.style.background = '#28a745';
    
    // Auto-hide after 3 seconds
    clearTimeout(notification.timeout);
    notification.timeout = setTimeout(() => {
        notification.style.animation = 'fadeOut 0.5s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 500);
    }, 3000);
}

function cleanupListeners() {
    console.log("🧹 Cleaning up listeners...");
    unsubscribeFunctions.forEach(unsubscribe => {
        try {
            unsubscribe();
        } catch (error) {
            console.error("Error unsubscribing:", error);
        }
    });
    unsubscribeFunctions = [];
    isListening = false;
}

function toggleAutoRefresh() {
    if (isListening) {
        cleanupListeners();
        showUpdateNotification("Updates paused. Click Refresh Data to update manually.");
    } else {
        loadUpgradeSummary();
    }
}

// Clean up listeners when page is unloaded
window.addEventListener('beforeunload', cleanupListeners);

// Make functions available globally
window.loadUpgradeSummary = loadUpgradeSummary;
window.toggleAutoRefresh = toggleAutoRefresh;

// Run on page load
document.addEventListener('DOMContentLoaded', loadUpgradeSummary);