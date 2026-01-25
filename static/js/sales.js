// sales.js - ONE-TAP BATCH-AWARE SALES SYSTEM (FIXED STOCK CHECKING LOGIC)
// EMERGENCY FIX: Added backend data mismatch handling

import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";
import { db } from "./firebase-config.js";
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

const FLASK_BACKEND_URL = window.location.origin;

// Global state
let salesOverlay = null;
let searchTimeout = null;
let currentShopId = null;
let currentUser = null;

const NAV_HEIGHT = 64;

// ====================================================
// FLOATING POINT PRECISION FIX
// ====================================================

function safeFloat(value) {
    // Fix floating point precision issues
    if (typeof value !== 'number') return 0;
    
    // Round to 10 decimal places to avoid floating point errors
    return Math.round(value * 10000000000) / 10000000000;
}

function safeCompare(a, b, threshold = 0.0000001) {
    // Compare numbers with tolerance for floating point errors
    return Math.abs(safeFloat(a) - safeFloat(b)) < threshold;
}

// ====================================================
// BATCH INTELLIGENCE - SEPARATE TRACKING FOR BASE VS SELLING UNITS
// ====================================================

class BatchIntelligence {
    constructor() {
        // SEPARATE tracking for base items vs selling units
        this.baseItemBatchState = new Map();   // item_id -> {currentBatchId, tapsCount}
        this.sellingUnitBatchState = new Map(); // item_id_sell_unit_id -> {currentBatchId, tapsCount}
    }
    
    getItemKey(item) {
        // DIFFERENT KEYS for base vs selling unit
        if (item.type === 'selling_unit') {
            return `${item.item_id}_${item.sell_unit_id}`;
        } else {
            return `${item.item_id}_main`;
        }
    }
    
    getBatchKey(item) {
        // DIFFERENT batch tracking for base vs selling unit
        if (item.type === 'selling_unit') {
            return `${item.item_id}_${item.sell_unit_id}_batch`;
        } else {
            return `${item.item_id}_main_batch`;
        }
    }
    
    prepareItemForCart(item) {
        const itemKey = this.getItemKey(item);
        const batchKey = this.getBatchKey(item);
        
        // Get appropriate state map
        const stateMap = item.type === 'selling_unit' 
            ? this.sellingUnitBatchState 
            : this.baseItemBatchState;
        
        // Get current batch state
        const currentState = stateMap.get(batchKey) || {
            currentBatchId: item.batch_id,
            tapsCount: 0,
            lastBatchId: item.batch_id
        };
        
        console.log(`📦 Batch analysis for ${item.name} (${item.type})`, {
            type: item.type,
            batchStatus: item.batch_status,
            currentBatch: item.batch_id,
            currentBatchRemaining: item.batch_remaining,
            safeCurrentBatchRemaining: safeFloat(item.batch_remaining || 0),
            nextBatchAvailable: item.next_batch_available,
            nextBatchRemaining: item.next_batch_remaining,
            safeNextBatchRemaining: safeFloat(item.next_batch_remaining || 0),
            nextBatchPrice: item.next_batch_price
        });
        
        // ====================================================
        // ⚠️ EMERGENCY FIX: Handle backend/frontend data mismatch
        // ====================================================
        
        // For BASE UNITS ONLY: If backend reports any issue, auto-switch proactively
        if ((item.type === 'base' || item.type === 'main_item')) {
            const currentStock = safeFloat(item.batch_remaining || 0);
            const hasNextBatch = item.next_batch_available;
            const nextStock = safeFloat(item.next_batch_remaining || 0);
            
            // ⚠️ CRITICAL: If batch_status indicates exhausted, force auto-switch
            if (item.batch_status === 'exhausted' && hasNextBatch && nextStock >= 0.999999) {
                console.log(`🚨 EMERGENCY: Backend reports batch exhausted, forcing auto-switch`);
                
                const switchedItem = {
                    ...item,
                    batch_id: item.next_batch_id,
                    batchId: item.next_batch_id,
                    price: item.next_batch_price,
                    batch_remaining: item.next_batch_remaining,
                    batch_name: item.next_batch_name,
                    next_batch_available: false,
                    next_batch_id: null,
                    next_batch_price: null,
                    next_batch_remaining: null,
                    next_batch_name: null,
                    _batch_switched: true,
                    _previous_batch_id: item.batch_id,
                    _previous_price: item.price,
                    _previous_stock: currentStock,
                    _emergency_switch: true
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
                    message: `Emergency auto-switch: Backend reports batch exhausted`
                };
            }
            
            // ⚠️ PROACTIVE SWITCH: If current batch shows low stock AND next batch is available, switch early
            if (currentStock < 2 && hasNextBatch && nextStock >= 0.999999) {
                console.log(`⚠️ PROACTIVE SWITCH: Current batch low (${currentStock}), next batch available (${nextStock})`);
                
                const switchedItem = {
                    ...item,
                    batch_id: item.next_batch_id,
                    batchId: item.next_batch_id,
                    price: item.next_batch_price,
                    batch_remaining: item.next_batch_remaining,
                    batch_name: item.next_batch_name,
                    next_batch_available: false,
                    next_batch_id: null,
                    next_batch_price: null,
                    next_batch_remaining: null,
                    next_batch_name: null,
                    _batch_switched: true,
                    _previous_batch_id: item.batch_id,
                    _previous_price: item.price,
                    _previous_stock: currentStock,
                    _proactive_switch: true
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
                    message: `Proactive auto-switch to prevent stock issues`
                };
            }
        }
        
        // ✅ CORRECT LOGIC: Check stock numbers directly WITH FLOATING POINT FIX
        
        // For SELLING UNITS: Just check if stock > 0 (with tolerance)
        if (item.type === 'selling_unit') {
            const stock = safeFloat(item.available_stock || item.batch_remaining || 0);
            if (stock <= 0.000001) { // Use tolerance for floating point
                console.log(`❌ Selling unit ${item.name} has no stock (${stock})`);
                return {
                    item: item,
                    action: 'cannot_add',
                    message: 'No stock available'
                };
            }
            
            // Normal case - add from current batch
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
        
        // For BASE UNITS: Complex stock checking WITH FLOATING POINT FIX
        if (item.type === 'base' || item.type === 'main_item') {
            const currentStock = safeFloat(item.batch_remaining || 0);
            
            // 1. Check if current batch has ≥ 1 unit (with tolerance)
            if (currentStock >= 0.999999) { // Use 0.999999 instead of 1
                console.log(`✅ Current batch has enough stock: ${currentStock} (≥ 0.999999)`);
                stateMap.set(batchKey, {
                    ...currentState,
                    tapsCount: currentState.tapsCount + 1
                });
                
                let action = 'add_current_batch';
                let message = '';
                
                if (currentStock < 1.999999) { // Last or almost last (with tolerance)
                    action = 'add_with_warning';
                    message = `Last item in ${item.batch_name || 'current batch'}!`;
                }
                
                return {
                    item: item,
                    action: action,
                    message: message
                };
            }
            
            // 2. Current batch < 1, check if next batch available with ≥ 1 unit
            if (currentStock < 0.999999 && item.next_batch_available) {
                const nextStock = safeFloat(item.next_batch_remaining || 0);
                
                if (nextStock >= 0.999999) {
                    console.log(`🔄 Auto-switch: Current batch ${currentStock}, Next batch ${nextStock}`);
                    
                    // Create switched item with new batch details
                    const switchedItem = {
                        ...item, // This copies ALL properties including category_id, category_name
                        batch_id: item.next_batch_id,
                        batchId: item.next_batch_id,
                        price: item.next_batch_price,
                        batch_remaining: item.next_batch_remaining,
                        batch_name: item.next_batch_name,
                        // Clear next batch info since we're switching to it
                        next_batch_available: false,
                        next_batch_id: null,
                        next_batch_price: null,
                        next_batch_remaining: null,
                        next_batch_name: null,
                        // Metadata for tracking
                        _batch_switched: true,
                        _previous_batch_id: item.batch_id,
                        _previous_price: item.price,
                        _previous_stock: currentStock
                    };
                    
                    // Update batch state
                    stateMap.set(batchKey, {
                        ...currentState,
                        currentBatchId: item.next_batch_id,
                        lastBatchId: item.batch_id,
                        tapsCount: 0 // Reset for new batch
                    });
                    
                    return {
                        item: switchedItem,
                        action: 'switch_and_add',
                        message: `Auto-switched to ${item.next_batch_name || 'new batch'} (${nextStock} units available)`
                    };
                } else {
                    console.log(`❌ Next batch also insufficient: ${nextStock} units (< 0.999999)`);
                }
            }
            
            // 3. Special case: Current batch is basically 0 due to floating point error
            if (currentStock < 0.000001 && item.next_batch_available) {
                console.log(`⚠️ Current batch effectively 0 (${currentStock}), checking next batch...`);
                const nextStock = safeFloat(item.next_batch_remaining || 0);
                if (nextStock >= 0.999999) {
                    console.log(`🔄 Auto-switch triggered for floating point error`);
                    // Same auto-switch logic as above
                    const switchedItem = {
                        ...item,
                        batch_id: item.next_batch_id,
                        batchId: item.next_batch_id,
                        price: item.next_batch_price,
                        batch_remaining: item.next_batch_remaining,
                        batch_name: item.next_batch_name,
                        next_batch_available: false,
                        next_batch_id: null,
                        next_batch_price: null,
                        next_batch_remaining: null,
                        next_batch_name: null,
                        _batch_switched: true,
                        _previous_batch_id: item.batch_id,
                        _previous_price: item.price,
                        _previous_stock: currentStock
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
                        message: `Auto-switched to ${item.next_batch_name || 'new batch'} (${nextStock} units available)`
                    };
                }
            }
            
            // 4. No batch with ≥ 1 unit available
            console.log(`❌ No batch with sufficient stock. Current: ${currentStock}`);
            return {
                item: item,
                action: 'cannot_add',
                message: 'Insufficient stock in any batch'
            };
        }
        
        // Fallback for unknown types
        console.warn(`Unknown item type: ${item.type}`);
        return {
            item: item,
            action: 'add_current_batch',
            message: ''
        };
    }
    
    // Clear batch tracking for a specific item
    clearItemTracking(item) {
        const batchKey = this.getBatchKey(item);
        const stateMap = item.type === 'selling_unit' 
            ? this.sellingUnitBatchState 
            : this.baseItemBatchState;
        stateMap.delete(batchKey);
    }
}

// Initialize batch intelligence
const batchIntelligence = new BatchIntelligence();

// ====================================================
// HELPER FUNCTIONS FOR ONE-TAP
// ====================================================

function getItemStock(item) {
    let stock = 0;
    
    if (item.type === 'selling_unit') {
        stock = item.available_stock || 0;
    } else {
        stock = item.batch_remaining || item.stock || 0;
    }
    
    return safeFloat(stock);
}

function getItemPrice(item) {
    return safeFloat(item.price || item.sellPrice || item.sell_price || 0);
}

function getStockColor(item) {
    const stock = getItemStock(item);
    
    // For selling units: any stock > 0 is good
    if (item.type === 'selling_unit') {
        if (stock > 0.000001) return '#2ed573'; // Green (with tolerance)
        return '#ff6b6b'; // Red
    }
    
    // For base units: check if can sell
    if (canAddToCart(item)) {
        if (stock >= 10) return '#2ed573';  // Green (good stock)
        if (stock >= 0.999999) return '#ffa502';   // Yellow (low but sellable, with tolerance)
        // Stock < 1 but can auto-switch
        if (item.next_batch_available && safeFloat(item.next_batch_remaining || 0) >= 0.999999) {
            return '#9b59b6'; // Purple (auto-switch ready)
        }
    }
    
    return '#ff6b6b'; // Red (cannot sell)
}

function getStockText(item) {
    const stock = getItemStock(item);
    
    if (item.type === 'selling_unit') {
        const unitName = item.display_name || item.name;
        if (stock > 0.000001) return `Available: ${stock.toFixed(6)} ${unitName}`;
        return '❌ Out of stock';
    }
    
    // For base units
    if (canAddToCart(item)) {
        if (stock >= 0.999999) {
            if (stock < 1.999999) return '🚨 Last item in batch!';
            return `Stock: ${stock.toFixed(2)}`;
        }
        
        // Stock < 1 but can auto-switch
        if (item.next_batch_available && safeFloat(item.next_batch_remaining || 0) >= 0.999999) {
            return `🔄 Auto-switch ready (${item.next_batch_remaining} available)`;
        }
    }
    
    return '❌ Out of stock';
}

function canAddToCart(item) {
    console.log(`🔍 Stock check for ${item.name} (${item.type}):`, {
        type: item.type,
        batch_remaining: item.batch_remaining,
        safe_batch_remaining: safeFloat(item.batch_remaining || 0),
        next_batch_available: item.next_batch_available,
        next_batch_remaining: item.next_batch_remaining,
        safe_next_batch_remaining: safeFloat(item.next_batch_remaining || 0),
        available_stock: item.available_stock,
        safe_available_stock: safeFloat(item.available_stock || 0),
        batch_status: item.batch_status  // Added for emergency fix
    });
    
    // For SELLING UNITS: Any stock > 0 (with floating point tolerance)
    if (item.type === 'selling_unit') {
        const stock = safeFloat(item.available_stock || item.batch_remaining || 0);
        // Use > 0.000001 instead of > 0 to handle floating point errors
        const canSell = stock > 0.000001;
        console.log(`📦 Selling unit check: ${stock} > 0.000001 = ${canSell}`);
        return canSell;
    }
    
    // For BASE UNITS: need ≥ 1 unit somewhere (with tolerance)
    const currentStock = safeFloat(item.batch_remaining || 0);
    
    // ⚠️ CRITICAL FIX: If batch_status says exhausted, check next batch even if frontend shows stock
    if (item.batch_status === 'exhausted' || item.batch_status === 'all_exhausted') {
        console.log(`⚠️ Batch status is '${item.batch_status}', checking next batch...`);
        if (item.next_batch_available) {
            const nextStock = safeFloat(item.next_batch_remaining || 0);
            if (nextStock >= 0.999999) {
                console.log(`🔄 Can add via next batch: ${nextStock} ≥ 0.999999`);
                return true;
            }
        }
        return false;
    }
    
    // 1. Current batch has ≥ 1 unit (with tolerance)
    if (currentStock >= 0.999999) {
        console.log(`✅ Current batch has ${currentStock} ≥ 0.999999`);
        return true;
    }
    
    // 2. Current batch < 1, but next batch has ≥ 1 unit (with tolerance)
    if (currentStock < 0.999999 && item.next_batch_available) {
        const nextStock = safeFloat(item.next_batch_remaining || 0);
        if (nextStock >= 0.999999) {
            console.log(`🔄 Next batch has ${nextStock} ≥ 0.999999`);
            return true;
        }
    }
    
    // 3. Special case: Current batch is basically 0 due to floating point error
    if (currentStock < 0.000001 && item.next_batch_available) {
        const nextStock = safeFloat(item.next_batch_remaining || 0);
        if (nextStock >= 0.999999) {
            console.log(`🔄 Floating point fix: current=${currentStock} (≈0), next=${nextStock} ≥ 0.999999`);
            return true;
        }
    }
    
    // 4. Cannot sell
    console.log(`❌ No stock available: current=${currentStock}, next_available=${item.next_batch_available}`);
    return false;
}

// ====================================================
// ONE-TAP ITEM HANDLER - FIXED WITH SEPARATE CART ENTRIES
// ====================================================

async function handleOneTap(item) {
    console.group(`ONE-TAP: ${item.name} (${item.type})`);
    console.log('Item received:', {
        type: item.type,
        item_id: item.item_id,
        sell_unit_id: item.sell_unit_id,
        batch_id: item.batch_id,
        price: item.price,
        safe_price: safeFloat(item.price || 0),
        batch_remaining: item.batch_remaining,
        safe_batch_remaining: safeFloat(item.batch_remaining || 0),
        next_batch_available: item.next_batch_available,
        next_batch_remaining: item.next_batch_remaining,
        safe_next_batch_remaining: safeFloat(item.next_batch_remaining || 0),
        batch_status: item.batch_status  // Added for debugging
    });
    
    // Debug: Check if we can add to cart BEFORE calling prepareItemForCart
    console.log('🔍 Pre-check canAddToCart:', canAddToCart(item));
    
    // Get batch intelligence decision
    const { item: cartItem, action, message } = batchIntelligence.prepareItemForCart(item);
    
    // Check if we can add to cart
    if (action === 'cannot_add') {
        console.log('❌ Cannot add to cart:', message);
        showNotification(message || 'Item out of stock!', 'error');
        console.groupEnd();
        return false;
    }
    
    // ✅ CRITICAL: Create UNIQUE cart entry ID
    const uniqueCartId = cartItem.type === 'selling_unit' 
        ? `${cartItem.item_id}_${cartItem.sell_unit_id}_${cartItem.batch_id}`
        : `${cartItem.item_id}_main_${cartItem.batch_id}`;
    
    // ENSURE ALL REQUIRED FIELDS ARE PRESENT
    const enrichedItem = {
        // ✅ UNIQUE ID for cart (differentiates base vs selling unit)
        id: uniqueCartId,
        cart_item_id: uniqueCartId,
        
        // Core IDs
        item_id: cartItem.item_id || item.item_id,
        main_item_id: cartItem.main_item_id || item.main_item_id || cartItem.item_id || item.item_id,
        
        // Names
        name: cartItem.name || item.name,
        display_name: cartItem.display_name || item.display_name || cartItem.name || item.name,
        
        // Quantity & Pricing
        quantity: 1, // Always 1 for one-tap
        sellPrice: cartItem.sellPrice || cartItem.sell_price || cartItem.price || 0,
        sell_price: cartItem.sellPrice || cartItem.sell_price || cartItem.price || 0,
        price: cartItem.price || cartItem.sellPrice || cartItem.sell_price || 0,
        
        // ✅ CRITICAL: CATEGORY FIELDS
        category_id: cartItem.category_id || item.category_id || 'unknown',
        category_name: cartItem.category_name || item.category_name || 'Uncategorized',
        
        // Stock
        stock: cartItem.stock || item.stock || cartItem.available_stock || item.available_stock || 0,
        available_stock: cartItem.available_stock || item.available_stock || cartItem.stock || item.stock || 0,
        
        // ✅ CRITICAL: TYPE MUST BE PRESERVED
        type: cartItem.type || item.type || 'main_item',
        
        // Batch Info
        batch_id: cartItem.batch_id || cartItem.batchId || item.batch_id || item.batchId || null,
        batchId: cartItem.batch_id || cartItem.batchId || item.batch_id || item.batchId || null,
        batch_name: cartItem.batch_name || item.batch_name || null,
        batch_remaining: cartItem.batch_remaining || item.batch_remaining || 0,
        
        // ✅ Selling Unit Info (only for selling units)
        sell_unit_id: cartItem.sell_unit_id || item.sell_unit_id || null,
        conversion_factor: cartItem.conversion_factor || item.conversion_factor || 1,
        
        // Batch Status (critical for emergency fix)
        batch_status: cartItem.batch_status || item.batch_status || 'unknown',
        
        // Optional
        thumbnail: cartItem.thumbnail || item.thumbnail || null,
        
        // Emergency fix metadata
        _emergency_switch: cartItem._emergency_switch || false,
        _proactive_switch: cartItem._proactive_switch || false,
        _batch_switched: cartItem._batch_switched || false
    };
    
    console.log('📦 Enriched item for cart:', {
        id: enrichedItem.id,
        type: enrichedItem.type,
        name: enrichedItem.name,
        price: enrichedItem.price,
        safe_price: safeFloat(enrichedItem.price || 0),
        batch_id: enrichedItem.batch_id,
        batch_remaining: enrichedItem.batch_remaining,
        safe_batch_remaining: safeFloat(enrichedItem.batch_remaining || 0),
        batch_status: enrichedItem.batch_status,
        action: action,
        emergency_switch: enrichedItem._emergency_switch,
        proactive_switch: enrichedItem._proactive_switch
    });
    
    // Show notification if needed
    if (message) {
        let notificationType = 'info';
        if (action === 'switch_and_add') {
            notificationType = enrichedItem._emergency_switch ? 'error' : 'warning';
        } else if (action === 'add_with_warning') {
            notificationType = 'warning';
        }
        
        showNotification(message, notificationType);
    }
    
    // Use cart-icon.js to add ONE item (one-tap = quantity 1)
    if (window.cartIcon && window.cartIcon.addItem) {
        console.log('🛒 Adding to cart via cart-icon.js (one-tap)', {
            name: enrichedItem.name,
            type: enrichedItem.type,
            unique_id: enrichedItem.id,
            batch_id: enrichedItem.batch_id,
            emergency_switch: enrichedItem._emergency_switch
        });
        
        // Pass the enriched item with UNIQUE ID
        const success = window.cartIcon.addItem(enrichedItem);
        
        if (success) {
            // Show success notification
            let successMsg = `Added 1 × ${item.name}`;
            if (action === 'switch_and_add') {
                if (enrichedItem._emergency_switch) {
                    successMsg += ` (Emergency batch switch)`;
                } else if (enrichedItem._proactive_switch) {
                    successMsg += ` (Proactive batch switch)`;
                } else {
                    successMsg += ` (Auto-switched to new batch)`;
                }
            }
            
            showNotification(successMsg, 'success', 2000);
            
            // Clear search for better UX
            const searchInput = document.getElementById('sales-search-input');
            const searchClear = document.getElementById('search-clear');
            if (searchInput) { 
                searchInput.value = ''; 
                searchInput.focus(); 
            }
            if (searchClear) searchClear.style.display = 'none';
            clearSearchResults();
            
            console.log('✅ Item added to cart successfully', { 
                type: enrichedItem.type,
                action: action,
                emergency_switch: enrichedItem._emergency_switch
            });
        } else {
            console.log('❌ Failed to add to cart');
            showNotification('Failed to add to cart', 'error');
        }
        
        console.groupEnd();
        return success;
    } else {
        console.log('❌ Cart system not loaded');
        showNotification('Cart system not ready. Please refresh.', 'error');
        console.groupEnd();
        return false;
    }
}

// ====================================================
// NOTIFICATION SYSTEM
// ====================================================

function showNotification(message, type = 'info', duration = 3000) {
    // Remove existing notification
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
    
    // Auto remove after duration
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
    
    // Add CSS animations if not already present
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
}

// ====================================================
// SALES OVERLAY (ONE-TAP VERSION)
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
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <div>
                    <h1 style="margin:0; color:white; font-size:26px; font-weight:700;">🛍️ One-Tap Sale</h1>
                    <p style="margin:6px 0 0; color:rgba(255,255,255,0.8); font-size:14px;">Tap once = 1 item added to cart</p>
                </div>
                <button id="close-sales" style="background: rgba(255,255,255,0.2); border:none; color:white; width:44px; height:44px; border-radius:12px; font-size:22px; cursor:pointer; flex-shrink:0;">×</button>
            </div>
            
            <!-- Search Box -->
            <div style="position:relative;">
                <div style="position:absolute; left:16px; top:50%; transform:translateY(-50%); color: rgba(255,255,255,0.7); font-size:18px; z-index:1;">🔍</div>
                <input id="sales-search-input" placeholder="Search products (type 2+ letters)..." style="width:100%; padding:16px 20px 16px 48px; border:none; border-radius:14px; font-size:16px; background: rgba(255,255,255,0.15); color:white; box-sizing:border-box;">
                <div id="search-clear" style="position:absolute; right:16px; top:50%; transform:translateY(-50%); color:rgba(255,255,255,0.7); font-size:20px; cursor:pointer; display:none; z-index:1;">×</div>
            </div>
            
            <!-- Batch Legend -->
            <div style="display:flex; gap:12px; margin-top:16px; flex-wrap:wrap;">
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:12px; height:12px; background:#2ed573; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:11px;">Good stock</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:12px; height:12px; background:#ff9f43; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:11px;">Last item</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:12px; height:12px; background:#ff6b6b; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:11px;">Out of stock</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:12px; height:12px; background:#9b59b6; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:11px;">Auto-switch ready</span>
                </div>
                <div style="display:flex; align-items:center; gap:4px;">
                    <div style="width:12px; height:12px; background:#e74c3c; border-radius:50%;"></div>
                    <span style="color:rgba(255,255,255,0.8); font-size:11px;">Emergency switch</span>
                </div>
            </div>
        </div>

        <!-- Results -->
        <div id="sales-results" style="flex:1; overflow-y:auto; padding:16px; -webkit-overflow-scrolling:touch;">
            <div style="text-align:center; color: rgba(255,255,255,0.6); padding:40px 20px;">
                <div style="font-size:56px; margin-bottom:16px; opacity:0.5;">🔍</div>
                <h3 style="margin:0 0 8px; color: rgba(255,255,255,0.9); font-size:18px;">Search Products</h3>
                <p style="margin:0; font-size:14px;">Type 2+ letters to search</p>
            </div>
        </div>
        
        <!-- Info Footer -->
        <div style="padding:12px 20px; background:rgba(0,0,0,0.2); color:rgba(255,255,255,0.7); font-size:12px; text-align:center;">
            👆 One tap = 1 item • System auto-switches batches when empty • Emergency fix active
        </div>
    `;

    document.body.appendChild(salesOverlay);

    // Event listeners
    document.getElementById("close-sales").onclick = closeSalesOverlay;
    const searchInput = document.getElementById("sales-search-input");
    const searchClear = document.getElementById("search-clear");

    searchInput.oninput = (e) => {
        const query = e.target.value;
        searchClear.style.display = query ? 'block' : 'none';
        onSearchInput(query);
    };

    searchInput.onkeydown = (e) => {
        if (e.key === 'Escape') {
            searchInput.value = '';
            searchClear.style.display = 'none';
            clearSearchResults();
        }
    };

    searchClear.onclick = () => {
        searchInput.value = '';
        searchClear.style.display = 'none';
        clearSearchResults();
        searchInput.focus();
    };
}

// ====================================================
// SEARCH FUNCTIONS
// ====================================================

function clearSearchResults() {
    const results = document.getElementById("sales-results");
    if (!results) return;
    results.innerHTML = `
        <div style="text-align:center; color: rgba(255,255,255,0.6); padding:40px 20px;">
            <div style="font-size:56px; margin-bottom:16px; opacity:0.5;">🔍</div>
            <h3 style="margin:0 0 8px; color: rgba(255,255,255,0.9); font-size:18px;">Search Products</h3>
            <p style="margin:0; font-size:14px;">Type 2+ letters to search</p>
        </div>
    `;
}

async function onSearchInput(query) {
    clearTimeout(searchTimeout);
    const results = document.getElementById("sales-results");

    if (!query.trim()) {
        clearSearchResults();
        return;
    }

    if (query.length < 2) {
        results.innerHTML = `<p style="text-align:center;color:white;padding:40px;">Type at least 2 letters to search...</p>`;
        return;
    }

    searchTimeout = setTimeout(async () => {
        console.log(`🔍 SEARCH: "${query}"`);
        
        results.innerHTML = `
            <div style="text-align:center; padding:40px 20px;">
                <div style="font-size:36px; margin-bottom:16px; color:rgba(255,255,255,0.7);">⏳</div>
                <h3 style="margin:0 0 8px; color: white; font-size:16px;">Searching for "${query}"</h3>
                <p style="margin:0; color:rgba(255,255,255,0.7); font-size:14px;">Checking inventory...</p>
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
                    user_id: currentUser?.uid 
                })
            });

            const data = await res.json();
            const searchTime = Date.now() - startTime;
            
            console.log(`✅ Search completed in ${searchTime}ms`, {
                results: data.items?.length || 0
            });
            
            if (!data.items?.length) {
                results.innerHTML = `
                    <div style="text-align:center; padding:40px 20px;">
                        <div style="font-size:36px; margin-bottom:16px; color:rgba(255,255,255,0.7);">🔍</div>
                        <h3 style="margin:0 0 8px; color: white; font-size:16px;">No items found</h3>
                        <p style="margin:0; color:rgba(255,255,255,0.7); font-size:14px;">Try a different search term</p>
                    </div>
                `;
                return;
            }
            
            renderResults(data.items);
            
        } catch (error) {
            console.log('❌ Search failed', error);
            
            results.innerHTML = `
                <div style="text-align:center; padding:40px 20px;">
                    <div style="font-size:36px; margin-bottom:16px; color:#ff6b6b;">❌</div>
                    <h3 style="margin:0 0 8px; color: white; font-size:16px;">Search failed</h3>
                    <p style="margin:0; color:rgba(255,255,255,0.7); font-size:14px;">Please try again</p>
                </div>
            `;
        }
    }, 150);
}

// ====================================================
// RENDER RESULTS WITH ONE-TAP FUNCTIONALITY
// ====================================================

function renderResults(items) {
    const resultsContainer = document.getElementById("sales-results");
    resultsContainer.innerHTML = '';
    
    console.log(`📋 Rendering ${items.length} results`);
    
    // Filter items based on canAddToCart
    const availableItems = items.filter(item => canAddToCart(item));
    const outOfStockItems = items.filter(item => !canAddToCart(item));
    
    console.log('📊 Item availability:', {
        total: items.length,
        available: availableItems.length,
        outOfStock: outOfStockItems.length
    });
    
    // Render available items first
    if (availableItems.length > 0) {
        const groupHeader = document.createElement('div');
        groupHeader.style.cssText = `
            color: rgba(255,255,255,0.9);
            font-size: 14px;
            font-weight: 600;
            margin: 20px 0 12px 0;
            padding-left: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        groupHeader.innerHTML = `✅ Available Items (${availableItems.length})`;
        resultsContainer.appendChild(groupHeader);
        
        availableItems.forEach(item => renderItemCard(item, resultsContainer));
    }
    
    // Render out of stock items
    if (outOfStockItems.length > 0) {
        const groupHeader = document.createElement('div');
        groupHeader.style.cssText = `
            color: rgba(255,255,255,0.7);
            font-size: 14px;
            font-weight: 600;
            margin: 20px 0 12px 0;
            padding-left: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        `;
        groupHeader.innerHTML = `❌ Out of Stock (${outOfStockItems.length})`;
        resultsContainer.appendChild(groupHeader);
        
        outOfStockItems.forEach(item => renderItemCard(item, resultsContainer));
    }
    
    // If no items at all
    if (items.length === 0) {
        resultsContainer.innerHTML = `
            <div style="text-align:center; color: rgba(255,255,255,0.6); padding:40px 20px;">
                <div style="font-size:56px; margin-bottom:16px; opacity:0.5;">🔍</div>
                <h3 style="margin:0 0 8px; color: rgba(255,255,255,0.9); font-size:18px;">No items found</h3>
                <p style="margin:0; font-size:14px;">Try a different search term</p>
            </div>
        `;
    }
    
    console.log(`✅ Rendered ${items.length} items`);
}

function renderItemCard(item, resultsContainer) {
    const stock = getItemStock(item);
    const stockColor = getStockColor(item);
    const stockText = getStockText(item);
    const canAdd = canAddToCart(item);
    const price = getItemPrice(item);
    
    // Determine batch indicator based on actual stock
    let batchIndicator = '';
    
    if (item.type === 'selling_unit') {
        if (stock > 0.000001) {
            batchIndicator = '📦 SELLING UNIT';
        } else {
            batchIndicator = '❌ OUT';
        }
    } else {
        // Base unit indicators
        if (canAdd) {
            if (stock >= 0.999999) {
                if (stock < 1.999999) {
                    batchIndicator = '🚨 LAST';
                } else if (stock < 10) {
                    batchIndicator = '⚠️ LOW';
                } else {
                    batchIndicator = '✅ IN STOCK';
                }
            } else if (stock < 0.999999 && item.next_batch_available && safeFloat(item.next_batch_remaining || 0) >= 0.999999) {
                batchIndicator = '🔄 AUTO-SWITCH';
            }
        } else {
            batchIndicator = '❌ OUT';
        }
    }
    
    // Emergency switch indicator
    if (item.batch_status === 'exhausted' && item.next_batch_available && safeFloat(item.next_batch_remaining || 0) >= 0.999999) {
        batchIndicator = '🚨 EMERGENCY SWITCH';
        stockColor = '#e74c3c';
    }
    
    const card = document.createElement('div');
    card.dataset.itemId = item.item_id;
    card.dataset.batchId = item.batch_id;
    card.dataset.canAdd = canAdd;
    
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

    let displayName = item.name;
    if (item.type === 'selling_unit' && item.display_name) {
        displayName = `${item.name.split('(')[0].trim()} (${item.display_name})`;
    }

    card.innerHTML = `
        ${batchIndicator ? `
            <div style="position:absolute; top:10px; right:10px; background:${stockColor}; color:white; padding:4px 10px; border-radius:10px; font-size:11px; font-weight:bold;">
                ${batchIndicator}
            </div>
        ` : ''}
        
        <div style="display:flex; align-items:center; gap:16px;">
            <div class="item-thumbnail" style="width:70px;height:70px;background:rgba(255,255,255,0.1);border-radius:12px;overflow:hidden;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                ${item.thumbnail ? 
                    `<img src="${item.thumbnail}" style="width:100%;height:100%;object-fit:cover;" onerror="this.style.display='none';this.parentElement.innerHTML='<span style=\\'font-size:26px;color:rgba(255,255,255,0.5)\\'>📦</span>';">` : 
                    `<span style="font-size:26px;color:rgba(255,255,255,0.5)">📦</span>`
                }
            </div>
            <div style="flex:1; min-width:0;">
                <div class="item-name" style="font-weight:600;color:${canAdd ? 'white' : 'rgba(255,255,255,0.6)'};font-size:16px;margin-bottom:6px;line-height:1.4;word-break:break-word;">${displayName}</div>
                
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
                
                <div style="color:${stockColor}; font-size:13px; font-weight:500; display:flex; align-items:center; gap:6px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:${stockColor};"></div>
                    ${stockText}
                </div>
                
                ${item.type === 'selling_unit' && item.conversion_factor ? 
                    `<div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:4px;">
                        1 Main Item = ${item.conversion_factor} ${item.display_name || 'units'}
                    </div>` : ''
                }
                
                ${item.next_batch_available ? 
                    `<div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:4px;">
                        Next batch: ${item.next_batch_name || 'Available'} (${item.next_batch_remaining || 0} units @ $${item.next_batch_price?.toFixed(2) || '???'})
                    </div>` : ''
                }
                
                ${item.batch_status === 'exhausted' ? 
                    `<div style="font-size:11px; color:#e74c3c; margin-top:4px; background:rgba(231,76,60,0.1); padding:4px 8px; border-radius:4px;">
                        🚨 Backend reports batch exhausted (will auto-switch)
                    </div>` : ''
                }
                
                ${!canAdd && item.type !== 'selling_unit' && stock > 0 && stock < 0.999999 ? 
                    `<div style="font-size:11px; color:#ff6b6b; margin-top:4px; background:rgba(255,107,107,0.1); padding:4px 8px; border-radius:4px;">
                        ⚠️ Current batch: ${stock.toFixed(6)} units (needs ≥ 1)
                    </div>` : ''
                }
            </div>
        </div>
    `;

    if (canAdd) {
        card.onclick = () => {
            console.log('Item selected:', {
                name: item.name,
                type: item.type,
                batch_id: item.batch_id,
                batch_remaining: item.batch_remaining,
                safe_batch_remaining: safeFloat(item.batch_remaining || 0),
                next_batch_available: item.next_batch_available,
                batch_status: item.batch_status
            });
            handleOneTap(item);
        };
        
        // Add click effect
        card.style.cursor = 'pointer';
        card.onmousedown = () => {
            card.style.transform = 'scale(0.98)';
        };
        card.onmouseup = () => {
            card.style.transform = 'scale(1)';
        };
    }
    
    resultsContainer.appendChild(card);
}

// ====================================================
// OPEN / CLOSE OVERLAY
// ====================================================

async function openSalesOverlay() {
    const auth = getAuth();
    currentUser = auth.currentUser;
    
    if (!currentUser) { 
        showNotification("Please login first", "error");
        return; 
    }
    
    console.log('🚀 Opening Sales Overlay');
    
    let shopId = currentUser.uid;
    try {
        const snap = await getDoc(doc(db, "Users", shopId));
        if (snap.exists() && snap.data().shop_id) {
            shopId = snap.data().shop_id;
            console.log('Shop ID resolved', { original: currentUser.uid, resolved: shopId });
        }
    } catch (error) {
        console.log('Error resolving shop ID', error);
    }
    
    currentShopId = shopId;
    console.log('Current shop ID set', { shopId });

    createSalesOverlay();
    salesOverlay.style.display = 'flex';
    
    console.log('Sales overlay displayed');
    
    setTimeout(() => {
        const input = document.getElementById("sales-search-input");
        if (input) {
            input.focus();
        }
    }, 50);
}

function closeSalesOverlay() {
    if (salesOverlay) {
        console.log('🔒 Closing Sales Overlay');
        salesOverlay.style.display = 'none';
    }
}

// ====================================================
// INITIALIZATION
// ====================================================

document.addEventListener("DOMContentLoaded", () => {
    console.log('⚡ Sales System Initialization');
    
    // Check if cart-icon.js is loaded
    if (!window.cartIcon) {
        console.log('⚠️ cart-icon.js not loaded yet. Waiting...');
        
        // Try to check again after a delay
        setTimeout(() => {
            if (window.cartIcon) {
                console.log('✅ cart-icon.js now loaded');
            } else {
                console.log('❌ cart-icon.js still not loaded');
                console.error('cart-icon.js is required for sales functionality');
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
            console.log('Sell button clicked');
            openSalesOverlay(); 
        });
        console.log('✅ Sell button initialized');
    } else {
        console.log('⚠️ Sell button not found in DOM');
    }
    
    // Add keyboard shortcut (Alt+S for sales)
    document.addEventListener('keydown', (e) => {
        if (e.altKey && e.key === 's') {
            e.preventDefault();
            console.log('Keyboard shortcut activated: Alt+S');
            openSalesOverlay();
        }
    });
    
    console.log('✅ Sales system ready');
    console.log('🚨 EMERGENCY FIX ACTIVE: Handling frontend/backend stock mismatch');
    
    console.log(`
╔═══════════════════════════════════════════╗
║     🛍️ ONE-TAP SALES SYSTEM READY        ║
╠═══════════════════════════════════════════╣
║ • One tap = 1 item to cart               ║
║ • Auto batch switching                   ║
║ • No quantity prompts                    ║
║ • Integrated with cart-icon.js           ║
║ • Press Alt+S to open sales              ║
║ • 🚨 EMERGENCY FIX: Frontend/Backend     ║
║   data mismatch handling                 ║
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
    handleOneTap
};