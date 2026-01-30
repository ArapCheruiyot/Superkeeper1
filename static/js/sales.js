// sales.js - SMART BATCH-AWARE SALES SYSTEM WITH BACKEND INTEGRATION
// UPDATED WITH WORKING CLOSE BUTTON AND BETTER EVENT HANDLING

import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";
import { db } from "./firebase-config.js";
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

const FLASK_BACKEND_URL = window.location.origin;

// Global state
let salesOverlay = null;
let searchTimeout = null;
let currentShopId = null;
let currentUser = null;
let currentCartId = null;
let currentSearchResults = []; // Store current search results
let currentSearchQuery = ""; // Store current search query

const NAV_HEIGHT = 64;

// ====================================================
// CART ID MANAGEMENT
// ====================================================

function getCurrentCartId() {
    if (!currentCartId) {
        currentCartId = localStorage.getItem('current_cart_id');
        if (!currentCartId) {
            currentCartId = 'cart_' + Date.now();
            localStorage.setItem('current_cart_id', currentCartId);
        }
    }
    return currentCartId;
}

// ====================================================
// FLOATING POINT PRECISION FIX
// ====================================================

function safeFloat(value) {
    if (typeof value !== 'number') return 0;
    return Math.round(value * 10000000000) / 10000000000;
}

// ====================================================
// BATCH INTELLIGENCE - UPDATED FOR SMART BACKEND
// ====================================================

class SmartBatchIntelligence {
    constructor() {
        this.baseItemBatchState = new Map();
        this.sellingUnitBatchState = new Map();
    }
    
    getBatchKey(item) {
        if (item.type === 'selling_unit') {
            return `${item.item_id}_${item.sell_unit_id}_batch`;
        } else {
            return `${item.item_id}_main_batch`;
        }
    }
    
    prepareItemForCart(item) {
        const batchKey = this.getBatchKey(item);
        const stateMap = item.type === 'selling_unit' 
            ? this.sellingUnitBatchState 
            : this.baseItemBatchState;
        
        const currentState = stateMap.get(batchKey) || {
            currentBatchId: item.batch_id,
            tapsCount: 0,
            lastBatchId: item.batch_id
        };
        
        console.log(`🧠 Smart batch analysis for ${item.name}:`, {
            type: item.type,
            can_fulfill: item.can_fulfill,
            batch_switch_required: item.batch_switch_required,
            real_available: item.real_available,
            notifications: item.notifications?.length || 0
        });
        
        // ✅ TRUST BACKEND'S DECISION FIRST
        if (item.can_fulfill === false && item.batch_switch_required && item.next_batch_id) {
            console.log(`🔄 Backend recommends batch switch to ${item.next_batch_id}`);
            
            const switchedItem = {
                ...item,
                batch_id: item.next_batch_id,
                batchId: item.next_batch_id,
                price: item.next_batch_price,
                batch_remaining: item.next_batch_remaining,
                batch_name: item.next_batch_name,
                next_batch_available: false,
                _batch_switched: true,
                _previous_batch_id: item.batch_id
            };
            
            stateMap.set(batchKey, {
                ...currentState,
                currentBatchId: item.next_batch_id,
                lastBatchId: item.batch_id,
                tapsCount: 0
            });
            
            return {
                item: switchedItem,
                action: 'switch_and_add',
                message: this.getSwitchMessage(item)
            };
        }
        
        // ✅ Backend says we can fulfill from current batch
        if (item.can_fulfill === true) {
            stateMap.set(batchKey, {
                ...currentState,
                tapsCount: currentState.tapsCount + 1
            });
            
            let action = 'add_current_batch';
            let message = '';
            
            // Check for low stock notifications
            if (item.notifications && item.notifications.length > 0) {
                const lowStockNotif = item.notifications.find(n => 
                    n.type === 'low_stock_warning' || n.severity === 'warning'
                );
                if (lowStockNotif) {
                    action = 'add_with_warning';
                    message = lowStockNotif.message;
                }
            }
            
            return {
                item: item,
                action: action,
                message: message
            };
        }
        
        // ⚠️ Fallback for old backend compatibility
        console.warn('⚠️ Using fallback logic - backend may be outdated');
        return this.fallbackPrepareItemForCart(item, stateMap, currentState);
    }
    
    getSwitchMessage(item) {
        if (item.notifications && item.notifications.length > 0) {
            const switchNotif = item.notifications.find(n => 
                n.type.includes('switch') || n.type.includes('insufficient')
            );
            if (switchNotif) return switchNotif.message;
        }
        
        return `Auto-switched to ${item.next_batch_name || 'next batch'} ($${item.next_batch_price?.toFixed(2) || item.price?.toFixed(2)})`;
    }
    
    fallbackPrepareItemForCart(item, stateMap, currentState) {
        // Keep old emergency logic as fallback only
        const stock = safeFloat(item.batch_remaining || 0);
        
        if (item.type === 'selling_unit') {
            if (stock <= 0.000001) {
                return {
                    item: item,
                    action: 'cannot_add',
                    message: 'No stock available'
                };
            }
            
            stateMap.set(batchKey, {
                ...currentState,
                tapsCount: currentState.tapsCount + 1
            });
            
            return {
                item: item,
                action: 'add_current_batch',
                message: ''
            };
        }
        
        // Base unit fallback logic
        if (stock >= 0.999999) {
            stateMap.set(batchKey, {
                ...currentState,
                tapsCount: currentState.tapsCount + 1
            });
            
            return {
                item: item,
                action: stock < 1.999999 ? 'add_with_warning' : 'add_current_batch',
                message: stock < 1.999999 ? 'Last item in batch!' : ''
            };
        }
        
        return {
            item: item,
            action: 'cannot_add',
            message: 'Insufficient stock'
        };
    }
    
    clearItemTracking(item) {
        const batchKey = this.getBatchKey(item);
        const stateMap = item.type === 'selling_unit' 
            ? this.sellingUnitBatchState 
            : this.baseItemBatchState;
        stateMap.delete(batchKey);
    }
}

// Initialize smart batch intelligence
const batchIntelligence = new SmartBatchIntelligence();

// ====================================================
// SMART STOCK CHECKING (USES BACKEND DATA)
// ====================================================

function canAddToCart(item) {
    // ✅ PRIMARY: Use backend's smart calculation
    if (item.can_fulfill !== undefined) {
        console.log(`✅ Backend can_fulfill: ${item.can_fulfill}`);
        return item.can_fulfill;
    }
    
    // ⚠️ FALLBACK: For backward compatibility
    console.log('⚠️ Using fallback stock check');
    const stock = item.real_available !== undefined ? item.real_available : item.batch_remaining;
    return safeFloat(stock) >= 0.999999;
}

function getStockColor(item) {
    // Check backend notifications first
    if (item.notifications && item.notifications.length > 0) {
        const urgentNotif = item.notifications.find(n => n.severity === 'error');
        if (urgentNotif) return '#e74c3c';
        
        const warningNotif = item.notifications.find(n => n.severity === 'warning');
        if (warningNotif) return '#ff9f43';
    }
    
    // Use backend's can_fulfill flag
    if (item.can_fulfill === false) {
        if (item.batch_switch_required && item.next_batch_available) {
            return '#9b59b6'; // Auto-switch ready
        }
        
        // Check for partial availability in selling units
        if (item.type === 'selling_unit' && item.real_available_fraction > 0) {
            return '#ff9f43'; // Partial stock - warning color
        }
        
        return '#ff6b6b'; // Cannot add
    }
    
    // Good stock
    const stock = item.real_available !== undefined ? item.real_available : item.batch_remaining;
    if (stock >= 10) return '#2ed573';
    if (stock >= 1) return '#ffa502';
    
    return '#ff6b6b';
}

function getStockText(item) {
    // Use backend notifications first
    if (item.notifications && item.notifications.length > 0) {
        const stockNotif = item.notifications.find(n => 
            n.type === 'low_stock_warning' || 
            n.type === 'insufficient_for_base' ||
            n.type === 'insufficient_for_selling_unit'
        );
        if (stockNotif) return stockNotif.message;
    }
    
    // Special handling for selling units with partial stock
    if (item.type === 'selling_unit' && item.real_available_fraction > 0) {
        const totalAvailable = (item.real_available_units || 0) + item.real_available_fraction;
        return `Partial: ${totalAvailable.toFixed(2)} units (needs ${item.conversion_factor || 1} for full unit)`;
    }
    
    // Fallback text
    const stock = item.real_available !== undefined ? item.real_available : item.batch_remaining;
    
    if (item.can_fulfill === false) {
        if (item.batch_switch_required && item.next_batch_available) {
            return '🔄 Auto-switch ready';
        }
        return '❌ Out of stock';
    }
    
    if (stock < 2) return '🚨 Low stock';
    return `Stock: ${stock.toFixed(2)}`;
}

function getItemPrice(item) {
    return safeFloat(item.price || item.sellPrice || item.sell_price || 0);
}

// ====================================================
// ONE-TAP ITEM HANDLER (UPDATED) - PERSISTENT SEARCH FIX
// ====================================================

async function handleOneTap(item) {
    console.group(`ONE-TAP: ${item.name} (${item.type})`);
    console.log('Smart item data:', {
        type: item.type,
        batch_id: item.batch_id,
        can_fulfill: item.can_fulfill,
        batch_switch_required: item.batch_switch_required,
        notifications: item.notifications,
        is_current_batch: item.is_current_batch,
        search_score: item.search_score || 0
    });
    
    const { item: cartItem, action, message } = batchIntelligence.prepareItemForCart(item);
    
    if (action === 'cannot_add') {
        console.log('❌ Cannot add to cart:', message);
        showNotification(message || 'Item out of stock!', 'error');
        console.groupEnd();
        return false;
    }
    
    // Create unique cart entry ID
    const uniqueCartId = cartItem.type === 'selling_unit' 
        ? `${cartItem.item_id}_${cartItem.sell_unit_id}_${cartItem.batch_id}`
        : `${cartItem.item_id}_main_${cartItem.batch_id}`;
    
    // Enrich item with all required fields
    const enrichedItem = {
        id: uniqueCartId,
        cart_item_id: uniqueCartId,
        item_id: cartItem.item_id,
        main_item_id: cartItem.main_item_id || cartItem.item_id,
        name: cartItem.name,
        display_name: cartItem.display_name || cartItem.name,
        quantity: 1,
        sellPrice: cartItem.price || cartItem.sellPrice || cartItem.sell_price || 0,
        price: cartItem.price || cartItem.sellPrice || cartItem.sell_price || 0,
        category_id: cartItem.category_id || 'unknown',
        category_name: cartItem.category_name || 'Uncategorized',
        type: cartItem.type || 'main_item',
        batch_id: cartItem.batch_id,
        batchId: cartItem.batch_id,
        batch_name: cartItem.batch_name,
        batch_remaining: cartItem.batch_remaining || cartItem.real_available || 0,
        sell_unit_id: cartItem.sell_unit_id,
        conversion_factor: cartItem.conversion_factor || 1,
        real_available: cartItem.real_available,
        can_fulfill: cartItem.can_fulfill,
        batch_switch_required: cartItem.batch_switch_required,
        is_current_batch: cartItem.is_current_batch,
        thumbnail: cartItem.thumbnail,
        _batch_switched: cartItem._batch_switched || false,
        search_score: cartItem.search_score || 0  // Preserve search score
    };
    
    console.log('📦 Enriched cart item:', {
        id: enrichedItem.id,
        type: enrichedItem.type,
        batch_id: enrichedItem.batch_id,
        can_fulfill: enrichedItem.can_fulfill,
        search_score: enrichedItem.search_score
    });
    
    // Show notification if needed
    if (message) {
        let notificationType = 'info';
        if (action === 'switch_and_add') notificationType = 'warning';
        if (action === 'add_with_warning') notificationType = 'warning';
        
        showNotification(message, notificationType);
    }
    
    // Add to cart via cart-icon.js
    if (window.cartIcon && window.cartIcon.addItem) {
        const success = window.cartIcon.addItem(enrichedItem);
        
        if (success) {
            let successMsg = `Added 1 × ${item.name}`;
            if (action === 'switch_and_add') {
                successMsg += ` (Auto-switched batch)`;
            }
            
            showNotification(successMsg, 'success', 2000);
            
            // ====================================================
            // FIX: DON'T CLEAR SEARCH RESULTS - JUST REFRESH THEM!
            // ====================================================
            console.log('🔄 Refreshing current search results (not clearing)');
            
            // Keep the search input focused
            const searchInput = document.getElementById('sales-search-input');
            if (searchInput) { 
                searchInput.focus(); 
            }
            
            // Refresh the displayed results to show updated stock/cart status
            if (currentSearchQuery && currentSearchResults.length > 0) {
                console.log('♻️ Refreshing displayed results for:', currentSearchQuery);
                renderEnhancedResults(currentSearchResults);
            }
            
            console.log('✅ Item added to cart successfully (search preserved)');
        } else {
            console.log('❌ Failed to add to cart');
            showNotification('Failed to add to cart', 'error');
        }
        
        console.groupEnd();
        return success;
    }
    
    console.log('❌ Cart system not loaded');
    showNotification('Cart system not ready', 'error');
    console.groupEnd();
    return false;
}

// ====================================================
// NOTIFICATION SYSTEM
// ====================================================

function showNotification(message, type = 'info', duration = 3000) {
    const existing = document.getElementById('sales-notification');
    if (existing) existing.remove();
    
    const colors = {
        info: { bg: '#3498db', icon: 'ℹ️' },
        success: { bg: '#2ecc71', icon: '✅' },
        warning: { bg: '#f39c12', icon: '⚠️' },
        error: { bg: '#e74c3c', icon: '🚨' }
    };
    
    const config = colors[type] || colors.info;
    
    const notification = document.createElement('div');
    notification.id = 'sales-notification';
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: ${config.bg};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 10px;
        max-width: 400px;
        animation: slideIn 0.3s ease;
    `;
    
    notification.innerHTML = `
        <span style="font-size: 18px;">${config.icon}</span>
        <span style="font-size: 14px; font-weight: 500;">${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// ====================================================
// SALES OVERLAY (WITH WORKING CLOSE BUTTON)
// ====================================================

function createSalesOverlay() {
    if (salesOverlay) return;

    salesOverlay = document.createElement("div");
    salesOverlay.id = "sales-overlay";
    salesOverlay.style.cssText = `
        position: fixed;
        top: ${NAV_HEIGHT}px;
        left: 0;
        width: 100%;
        height: calc(100vh - ${NAV_HEIGHT}px);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        z-index: 2000;
        display: none;
        flex-direction: column;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        overflow: hidden;
    `;

    salesOverlay.innerHTML = `
        <!-- Header -->
        <div style="padding: 20px; background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.2); flex-shrink:0;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; position:relative;">
                <div>
                    <h1 style="margin:0; color:white; font-size:26px; font-weight:700;">🛍️ Smart Sales</h1>
                    <p style="margin:6px 0 0; color:rgba(255,255,255,0.8); font-size:14px;">Smart batch switching • Enhanced search</p>
                </div>
                <!-- CLOSE BUTTON MOVED LOWER: Added margin-top: 8px -->
                <button id="close-sales-overlay-btn" style="background: rgba(255,255,255,0.2); border:none; color:white; width:44px; height:44px; border-radius:12px; font-size:22px; cursor:pointer; flex-shrink:0; margin-top: 8px;">×</button>
            </div>
            
            <!-- Search Box -->
            <div style="position:relative;">
                <div style="position:absolute; left:16px; top:50%; transform:translateY(-50%); color: rgba(255,255,255,0.7); font-size:18px; z-index:1;">🔍</div>
                <input id="sales-search-input" placeholder="Search products..." style="width:100%; padding:16px 20px 16px 48px; border:none; border-radius:14px; font-size:16px; background: rgba(255,255,255,0.15); color:white; box-sizing:border-box;">
                <div id="search-clear" style="position:absolute; right:16px; top:50%; transform:translateY(-50%); color:rgba(255,255,255,0.7); font-size:20px; cursor:pointer; display:none; z-index:1;">×</div>
            </div>
            
            <!-- Smart Legend -->
            <div style="display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; justify-content:center;">
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:10px; height:10px; background:#2ed573; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:10px;">Available</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:10px; height:10px; background:#ff9f43; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:10px;">Low stock</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:10px; height:10px; background:#ff6b6b; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:10px;">Out of stock</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:10px; height:10px; background:#9b59b6; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:10px;">Auto-switch</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:10px; height:10px; background:#3498db; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:10px;">In your cart</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:10px; height:10px; background:#ff9f43; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:10px;">Partial stock</span>
                </div>
            </div>
        </div>

        <!-- Results -->
        <div id="sales-results" style="flex:1; overflow-y:auto; padding:16px; -webkit-overflow-scrolling:touch;">
            <div style="text-align:center; color: rgba(255,255,255,0.6); padding:40px 20px;">
                <div style="font-size:56px; margin-bottom:16px; opacity:0.5;">🔍</div>
                <h3 style="margin:0 0 8px; color: rgba(255,255,255,0.9); font-size:18px;">Enhanced Search</h3>
                <p style="margin:0; font-size:14px;">Type to search with smart scoring</p>
            </div>
        </div>
        
        <!-- Info Footer -->
        <div style="padding:12px 20px; background:rgba(0,0,0,0.2); color:rgba(255,255,255,0.7); font-size:12px; text-align:center;">
            👆 One tap • Smart batch switching • Enhanced search relevance
        </div>
    `;

    document.body.appendChild(salesOverlay);

    // IMPORTANT: Set up event listeners with proper timing
    setTimeout(() => {
        setupOverlayEventListeners();
    }, 50);
}

// ====================================================
// OVERLAY EVENT LISTENERS (SEPARATED FOR BETTER TIMING)
// ====================================================

function setupOverlayEventListeners() {
    console.log('⚡ Setting up overlay event listeners');
    
    // Fix 1: Use event delegation for close button
    const closeBtn = document.getElementById('close-sales-overlay-btn');
    if (closeBtn) {
        console.log('✅ Found close button, attaching event listener');
        closeBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('🔘 Close button clicked');
            closeSalesOverlay();
        };
        
        // Also add pointer-events check
        closeBtn.style.pointerEvents = 'auto';
        closeBtn.style.cursor = 'pointer';
    } else {
        console.log('❌ Close button not found, will retry...');
        // Retry after a short delay
        setTimeout(() => {
            const retryBtn = document.getElementById('close-sales-overlay-btn');
            if (retryBtn) {
                console.log('✅ Found close button on retry');
                retryBtn.onclick = closeSalesOverlay;
                retryBtn.style.pointerEvents = 'auto';
                retryBtn.style.cursor = 'pointer';
            }
        }, 100);
    }
    
    // Search input event listeners
    const searchInput = document.getElementById("sales-search-input");
    const searchClear = document.getElementById("search-clear");
    
    if (searchInput) {
        searchInput.oninput = (e) => {
            const query = e.target.value;
            if (searchClear) {
                searchClear.style.display = query ? 'block' : 'none';
            }
            onSearchInput(query);
        };

        searchInput.onkeydown = (e) => {
            if (e.key === 'Escape') {
                searchInput.value = '';
                if (searchClear) {
                    searchClear.style.display = 'none';
                }
                clearSearchResults();
            }
        };
    }
    
    if (searchClear) {
        searchClear.onclick = () => {
            if (searchInput) {
                searchInput.value = '';
                searchClear.style.display = 'none';
                clearSearchResults();
                searchInput.focus();
            }
        };
    }
    
    // Add global click handler for the overlay background
    salesOverlay.addEventListener('click', function(e) {
        // If click is on the overlay background (not on content), close it
        if (e.target === salesOverlay) {
            closeSalesOverlay();
        }
    });
}

// ====================================================
// SEARCH FUNCTIONS (UPDATED WITH CART ID)
// ====================================================

function clearSearchResults() {
    const results = document.getElementById("sales-results");
    if (!results) return;
    results.innerHTML = `
        <div style="text-align:center; color: rgba(255,255,255,0.6); padding:40px 20px;">
            <div style="font-size:56px; margin-bottom:16px; opacity:0.5;">🔍</div>
            <h3 style="margin:0 0 8px; color: rgba(255,255,255,0.9); font-size:18px;">Enhanced Search</h3>
            <p style="margin:0; font-size:14px;">Type to search with smart scoring</p>
        </div>
    `;
    // Also clear stored results
    currentSearchResults = [];
    currentSearchQuery = "";
}

async function onSearchInput(query) {
    clearTimeout(searchTimeout);
    const results = document.getElementById("sales-results");

    if (!query.trim()) {
        clearSearchResults();
        return;
    }

    if (query.length < 2) {
        results.innerHTML = `<p style="text-align:center;color:white;padding:40px;">Type at least 2 letters...</p>`;
        return;
    }

    searchTimeout = setTimeout(async () => {
        console.log(`🔍 ENHANCED SEARCH: "${query}"`);
        
        // Store the current search query
        currentSearchQuery = query;
        
        results.innerHTML = `
            <div style="text-align:center; padding:40px 20px;">
                <div style="font-size:36px; margin-bottom:16px; color:rgba(255,255,255,0.7);">⚡</div>
                <h3 style="margin:0 0 8px; color: white; font-size:16px;">Enhanced search for "${query}"</h3>
                <p style="margin:0; color:rgba(255,255,255,0.7); font-size:14px;">Finding best matches...</p>
            </div>
        `;
        
        try {
            const startTime = Date.now();
            
            const res = await fetch(`${FLASK_BACKEND_URL}/sales`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    query, 
                    shop_id: currentShopId,
                    cart_id: getCurrentCartId(),
                    user_id: currentUser?.uid 
                })
            });

            const data = await res.json();
            const searchTime = Date.now() - startTime;
            
            console.log(`✅ Enhanced search completed in ${searchTime}ms`, {
                results: data.items?.length || 0,
                scored_results: data.meta?.scored_results || 0,
                high_score_results: data.meta?.high_score_results || 0,
                main_items_count: data.meta?.main_items_count || 0,
                selling_units_count: data.meta?.selling_units_count || 0
            });
            
            // Log search scores for debugging
            if (data.items && data.items.length > 0) {
                console.log('🔍 Search scores:', data.items.map(item => ({
                    name: item.name,
                    type: item.type,
                    score: item.search_score || 0,
                    can_fulfill: item.can_fulfill,
                    real_available_fraction: item.real_available_fraction || 0
                })));
            }
            
            if (!data.items?.length) {
                results.innerHTML = `
                    <div style="text-align:center; padding:40px 20px;">
                        <div style="font-size:36px; margin-bottom:16px; color:rgba(255,255,255,0.7);">🔍</div>
                        <h3 style="margin:0 0 8px; color: white; font-size:16px;">No items found</h3>
                        <p style="margin:0; color:rgba(255,255,255,0.7); font-size:14px;">Try a different search term</p>
                    </div>
                `;
                // Clear stored results
                currentSearchResults = [];
                return;
            }
            
            // Store the results for later refresh
            currentSearchResults = data.items;
            renderEnhancedResults(data.items);
            
        } catch (error) {
            console.log('❌ Search failed', error);
            
            results.innerHTML = `
                <div style="text-align:center; padding:40px 20px;">
                    <div style="font-size:36px; margin-bottom:16px; color:#ff6b6b;">❌</div>
                    <h3 style="margin:0 0 8px; color: white; font-size:16px;">Search failed</h3>
                    <p style="margin:0; color:rgba(255,255,255,0.7); font-size:14px;">${error.message}</p>
                </div>
            `;
            // Clear stored results on error
            currentSearchResults = [];
        }
    }, 150);
}

// ====================================================
// ENHANCED RESULTS RENDERING WITH SEARCH SCORES
// ====================================================

function renderEnhancedResults(items) {
    const resultsContainer = document.getElementById("sales-results");
    resultsContainer.innerHTML = '';
    
    console.log(`📋 Rendering ${items.length} enhanced results`);
    
    // NEW: Better grouping logic that handles partial stock
    const bestMatches = [];
    const goodMatches = [];
    const lowStockMatches = [];
    const partialStockMatches = [];
    const unavailableMatches = [];
    
    items.forEach(item => {
        const score = item.search_score || 0;
        const canFulfill = item.can_fulfill !== false;
        
        // Check if it's a selling unit with partial stock
        const isSellingUnit = item.type === 'selling_unit';
        const hasPartialStock = item.real_available_fraction > 0;
        const needsBatchSwitch = item.batch_switch_required;
        const hasNextBatch = item.next_batch_available;
        
        // Grouping logic:
        // 1. High score + can fulfill = Best Matches
        // 2. High score + auto-switch available = Good Matches
        // 3. High score + partial stock = Partial Stock
        // 4. Medium score + can fulfill = Good Matches
        // 5. Low score or no stock = Unavailable
        
        if (score >= 80 && canFulfill) {
            bestMatches.push(item);
        } 
        else if (score >= 80 && !canFulfill && needsBatchSwitch && hasNextBatch) {
            // High score items that can auto-switch
            goodMatches.push(item);
        }
        else if (score >= 80 && !canFulfill && isSellingUnit && hasPartialStock) {
            // Selling units with partial stock (like "Ram stick 2gb")
            partialStockMatches.push(item);
        }
        else if (score >= 50 && canFulfill) {
            goodMatches.push(item);
        }
        else if (score >= 30 && !canFulfill && isSellingUnit && hasPartialStock) {
            // Selling units with partial stock but lower score
            partialStockMatches.push(item);
        }
        else if (score >= 30 && !canFulfill && item.batch_remaining > 0) {
            // Items with some stock but not enough to fulfill
            lowStockMatches.push(item);
        }
        else if (score > 0) {
            unavailableMatches.push(item);
        }
    });
    
    // Helper function to render group header
    function renderGroupHeader(title, count, color = 'rgba(255,255,255,0.9)') {
        const header = document.createElement('div');
        header.style.cssText = `
            color: ${color};
            font-size: 14px;
            font-weight: 600;
            margin: 20px 0 12px 0;
            padding-left: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        header.innerHTML = `${title} (${count})`;
        return header;
    }
    
    console.log('📊 Enhanced search grouping:', {
        bestMatches: bestMatches.length,
        goodMatches: goodMatches.length,
        partialStockMatches: partialStockMatches.length,
        lowStockMatches: lowStockMatches.length,
        unavailableMatches: unavailableMatches.length
    });
    
    // Render groups in order of importance
    if (bestMatches.length > 0) {
        resultsContainer.appendChild(renderGroupHeader('🎯 Best Matches', bestMatches.length, '#2ed573'));
        bestMatches.forEach(item => renderEnhancedItemCard(item, resultsContainer, 'high'));
    }
    
    if (goodMatches.length > 0) {
        resultsContainer.appendChild(renderGroupHeader('✅ Good Matches', goodMatches.length, '#ffa502'));
        goodMatches.forEach(item => renderEnhancedItemCard(item, resultsContainer, 'medium'));
    }
    
    if (partialStockMatches.length > 0) {
        resultsContainer.appendChild(renderGroupHeader('⚠️ Partial Stock', partialStockMatches.length, '#ff9f43'));
        partialStockMatches.forEach(item => renderEnhancedItemCard(item, resultsContainer, 'partial'));
    }
    
    if (lowStockMatches.length > 0) {
        resultsContainer.appendChild(renderGroupHeader('📉 Low Stock', lowStockMatches.length, '#ff6b6b'));
        lowStockMatches.forEach(item => renderEnhancedItemCard(item, resultsContainer, 'low'));
    }
    
    if (unavailableMatches.length > 0) {
        resultsContainer.appendChild(renderGroupHeader('❌ Currently Unavailable', unavailableMatches.length, 'rgba(255,255,255,0.7)'));
        unavailableMatches.forEach(item => renderEnhancedItemCard(item, resultsContainer, 'unavailable'));
    }
    
    // Empty state
    if (items.length === 0) {
        resultsContainer.innerHTML = `
            <div style="text-align:center; color: rgba(255,255,255,0.6); padding:40px 20px;">
                <div style="font-size:56px; margin-bottom:16px; opacity:0.5;">🔍</div>
                <h3 style="margin:0 0 8px; color: rgba(255,255,255,0.9); font-size:18px;">No items found</h3>
                <p style="margin:0; font-size:14px;">Try a different search term</p>
            </div>
        `;
    }
}

function renderEnhancedItemCard(item, resultsContainer, matchQuality = 'medium') {
    const canAdd = canAddToCart(item);
    const stockColor = getStockColor(item);
    const stockText = getStockText(item);
    const price = getItemPrice(item);
    const searchScore = item.search_score || 0;
    
    // Determine batch indicator
    let batchIndicator = '';
    if (item.is_current_batch) {
        batchIndicator = '🛒 IN CART';
    } else if (item.can_fulfill === false && item.batch_switch_required) {
        batchIndicator = '🔄 AUTO-SWITCH';
    } else if (item.can_fulfill === true) {
        const stock = item.real_available !== undefined ? item.real_available : item.batch_remaining;
        if (stock < 2) {
            batchIndicator = '⚠️ LOW';
        } else {
            batchIndicator = '✅ AVAILABLE';
        }
    } else if (item.type === 'selling_unit' && item.real_available_fraction > 0) {
        batchIndicator = '⚠️ PARTIAL';
    } else {
        batchIndicator = '❌ UNAVAILABLE';
    }
    
    // Determine match quality indicator
    let matchIndicator = '';
    let matchColor = '#666';
    if (matchQuality === 'high') {
        matchIndicator = '🎯';
        matchColor = '#2ed573';
    } else if (matchQuality === 'medium') {
        matchIndicator = '✅';
        matchColor = '#ffa502';
    } else if (matchQuality === 'partial') {
        matchIndicator = '⚠️';
        matchColor = '#ff9f43';
    } else if (matchQuality === 'low') {
        matchIndicator = '📉';
        matchColor = '#ff6b6b';
    } else if (matchQuality === 'unavailable') {
        matchIndicator = '❌';
        matchColor = 'rgba(255,255,255,0.5)';
    }
    
    // Prepare display name
    let displayName = item.name;
    if (item.type === 'selling_unit' && item.display_name) {
        displayName = item.display_name;
    }
    
    const card = document.createElement('div');
    card.dataset.itemId = item.item_id;
    card.dataset.batchId = item.batch_id;
    card.dataset.canAdd = canAdd;
    card.dataset.searchScore = searchScore;
    
    card.style.cssText = `
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.1);
        cursor: ${canAdd ? 'pointer' : 'not-allowed'};
        position: relative;
        transition: transform 0.2s, box-shadow 0.2s;
        opacity: ${canAdd ? '1' : '0.7'};
        border-left: 4px solid ${matchColor};
    `;
    
    if (canAdd) {
        card.onmouseenter = () => {
            card.style.transform = 'translateY(-2px)';
            card.style.boxShadow = '0 8px 20px rgba(0,0,0,0.2)';
        };
        card.onmouseleave = () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = 'none';
        };
    }
    
    // Render notifications
    const notificationsHTML = item.notifications ? item.notifications.map(notif => `
        <div style="
            font-size: 11px;
            color: ${notif.severity === 'error' ? '#e74c3c' : 
                    notif.severity === 'warning' ? '#ff9f43' : '#3498db'};
            margin-top: 4px;
            background: ${notif.severity === 'error' ? 'rgba(231,76,60,0.1)' :
                        notif.severity === 'warning' ? 'rgba(255,159,67,0.1)' : 'rgba(52,152,219,0.1)'};
            padding: 4px 8px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
        ">
            ${notif.severity === 'error' ? '🚨' :
              notif.severity === 'warning' ? '⚠️' : 'ℹ️'}
            ${notif.message}
        </div>
    `).join('') : '';

    card.innerHTML = `
        ${batchIndicator ? `
            <div style="position:absolute; top:10px; right:10px; background:${stockColor}; color:white; padding:4px 10px; border-radius:10px; font-size:11px; font-weight:bold;">
                ${batchIndicator}
            </div>
        ` : ''}
        
        <!-- Match quality indicator -->
        <div style="position:absolute; top:10px; left:10px; background:${matchColor}; color:white; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px;">
            ${matchIndicator}
        </div>
        
        <div style="display:flex; align-items:center; gap:16px; margin-left:8px;">
            <div class="item-thumbnail" style="width:70px;height:70px;background:rgba(255,255,255,0.1);border-radius:12px;overflow:hidden;display:flex;align-items:center;justify-content:center;flex-shrink:0; margin-left:20px;">
                ${item.thumbnail ? 
                    `<img src="${item.thumbnail}" style="width:100%;height:100%;object-fit:cover;" onerror="this.style.display='none';this.parentElement.innerHTML='<span style=\\'font-size:26px;color:rgba(255,255,255,0.5)\\'>📦</span>';">` : 
                    `<span style="font-size:26px;color:rgba(255,255,255,0.5)">📦</span>`
                }
            </div>
            <div style="flex:1; min-width:0;">
                <div class="item-name" style="font-weight:600;color:${canAdd ? 'white' : 'rgba(255,255,255,0.6)'};font-size:16px;margin-bottom:6px;line-height:1.4;word-break:break-word;">
                    ${displayName}
                    ${item.is_current_batch ? '<span style="background:#3498db;color:white;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">IN CART</span>' : ''}
                    ${item.type === 'selling_unit' && item.parent_item_name ? 
                        `<span style="font-size:12px; color:rgba(255,255,255,0.7); margin-left:8px;">(${item.parent_item_name})</span>` : 
                        ''
                    }
                </div>
                
                <!-- Search score indicator (debug mode) -->
                <div style="font-size:10px; color:${matchColor}; margin-bottom:4px; display:flex; align-items:center; gap:4px;">
                    <span>Match: ${searchScore}/100</span>
                    ${item.matched_by ? `<span style="background:${matchColor}20; padding:1px 4px; border-radius:3px;">${item.matched_by}</span>` : ''}
                </div>
                
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px; flex-wrap:wrap;">
                    <div class="item-price" style="color:${canAdd ? '#ffd700' : 'rgba(255,215,0,0.6)'};font-weight:700;font-size:20px;flex-shrink:0;">
                        $${price.toFixed(2)}
                    </div>
                    ${item.batch_name ? `
                        <div style="background:rgba(255,255,255,0.1); color:rgba(255,255,255,0.8); padding:4px 8px; border-radius:6px; font-size:11px;">
                            ${item.batch_name}
                        </div>
                    ` : ''}
                    ${item.type === 'selling_unit' ? `
                        <div style="background:rgba(155,89,182,0.3); color:${canAdd ? 'white' : 'rgba(255,255,255,0.6)'}; padding:2px 6px; border-radius:4px; font-size:10px;">
                            Selling Unit
                        </div>
                    ` : ''}
                </div>
                
                <div style="color:${stockColor}; font-size:13px; font-weight:500; display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:${stockColor};"></div>
                    ${stockText}
                </div>
                
                ${item.type === 'selling_unit' && item.conversion_factor ? 
                    `<div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:4px;">
                        1 Main Item = ${item.conversion_factor} ${item.display_name || 'units'}
                    </div>` : ''
                }
                
                ${item.next_batch_available && item.can_fulfill === false ? 
                    `<div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:4px;">
                        Next: ${item.next_batch_name || 'batch'} (${item.next_batch_remaining || 0} @ $${item.next_batch_price?.toFixed(2) || price.toFixed(2)})
                    </div>` : ''
                }
                
                ${notificationsHTML}
            </div>
        </div>
    `;

    if (canAdd) {
        card.onclick = () => {
            console.log('Item selected for one-tap:', {
                name: item.name,
                type: item.type,
                batch_id: item.batch_id,
                can_fulfill: item.can_fulfill,
                search_score: searchScore
            });
            handleOneTap(item);
        };
        
        card.style.cursor = 'pointer';
        card.onmousedown = () => card.style.transform = 'scale(0.98)';
        card.onmouseup = () => card.style.transform = 'scale(1)';
    }
    
    resultsContainer.appendChild(card);
}

// ====================================================
// OPEN / CLOSE OVERLAY (UPDATED)
// ====================================================

async function openSalesOverlay() {
    const auth = getAuth();
    currentUser = auth.currentUser;
    
    if (!currentUser) { 
        showNotification("Please login first", "error");
        return; 
    }
    
    console.log('🚀 Opening Enhanced Sales Overlay');
    
    let shopId = currentUser.uid;
    try {
        const snap = await getDoc(doc(db, "Users", shopId));
        if (snap.exists() && snap.data().shop_id) {
            shopId = snap.data().shop_id;
            console.log('Shop ID resolved', { resolved: shopId });
        }
    } catch (error) {
        console.log('Error resolving shop ID', error);
    }
    
    currentShopId = shopId;
    console.log('Current shop ID set', { shopId });

    createSalesOverlay();
    salesOverlay.style.display = 'flex';
    
    // Clear any previous search when opening
    currentSearchResults = [];
    currentSearchQuery = "";
    
    // Setup event listeners with a slight delay to ensure DOM is ready
    setTimeout(() => {
        setupOverlayEventListeners();
        
        // Focus the search input
        const input = document.getElementById("sales-search-input");
        if (input) {
            input.focus();
            console.log('✅ Search input focused');
        }
    }, 100);
}

function closeSalesOverlay() {
    if (salesOverlay) {
        console.log('🔒 Closing Sales Overlay');
        salesOverlay.style.display = 'none';
        // Clear search state when closing
        currentSearchResults = [];
        currentSearchQuery = "";
        
        // Show a quick notification that overlay is closed
        showNotification('Sales overlay closed', 'info', 1000);
    }
}

// ====================================================
// INITIALIZATION
// ====================================================

document.addEventListener("DOMContentLoaded", () => {
    console.log('⚡ Enhanced Sales System Initialization');
    
    // Ensure cart ID exists
    getCurrentCartId();
    
    // Check cart system
    if (!window.cartIcon) {
        console.log('⚠️ cart-icon.js not loaded yet');
        setTimeout(() => {
            if (window.cartIcon) {
                console.log('✅ cart-icon.js now loaded');
            }
        }, 1000);
    } else {
        console.log('✅ cart-icon.js is loaded');
    }
    
    // Expose functions globally
    window.openSalesOverlay = openSalesOverlay;
    window.closeSalesOverlay = closeSalesOverlay;
    window.batchIntelligence = batchIntelligence;
    
    // Initialize sell button
    const sellBtn = document.getElementById("sell-btn");
    if (sellBtn) {
        sellBtn.addEventListener("click", e => { 
            e.preventDefault(); 
            openSalesOverlay(); 
        });
        console.log('✅ Sell button initialized');
    }
    
    // Keyboard shortcut
    document.addEventListener('keydown', (e) => {
        if (e.altKey && e.key === 's') {
            e.preventDefault();
            openSalesOverlay();
        }
        // Also allow ESC key to close overlay if it's open
        if (e.key === 'Escape' && salesOverlay && salesOverlay.style.display === 'flex') {
            closeSalesOverlay();
        }
    });
    
    console.log(`
╔═══════════════════════════════════════════╗
║     🎯 ENHANCED SALES SYSTEM READY       ║
╠═══════════════════════════════════════════╣
║ • Enhanced search scoring                ║
║ • Smart batch switching                  ║
║ • Cart-aware search                      ║
║ • Match quality indicators               ║
║ • Partial stock handling                 ║
║ • PERSISTENT SEARCH RESULTS              ║
║ • Working close button (✓)               ║
║ • Press Alt+S to open, ESC to close      ║
╚═══════════════════════════════════════════╝
`);
});

// ====================================================
// EXPORT FOR MODULE USAGE
// ====================================================

export {
    openSalesOverlay,
    closeSalesOverlay,
    batchIntelligence,
    handleOneTap,
    getCurrentCartId
};