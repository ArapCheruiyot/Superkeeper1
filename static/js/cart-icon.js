// cart-icon.js - FIXED WITH SEPARATE BASE/SELLING UNITS SUPPORT
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";
import { db } from "./firebase-config.js";
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

const FLASK_BACKEND_URL = window.location.origin;

// ====================================================
// GLOBAL CART STATE
// ====================================================
let cart = [];
let currentShopId = null;

// ====================================================
// DEBUG UTILITIES
// ====================================================

function debugLog(message, data = null) {
    console.log(`🛒 ${message}`, data || '');
}

function debugError(message, error = null) {
    console.error(`🛒 ❌ ${message}`, error || '');
}

// ====================================================
// CART MANAGEMENT FUNCTIONS
// ====================================================

function saveCartToStorage() {
    localStorage.setItem('sales_cart', JSON.stringify(cart));
    debugLog('Cart saved to storage', cart.length);
}

function loadCartFromStorage() {
    const saved = localStorage.getItem('sales_cart');
    if (saved) {
        try {
            cart = JSON.parse(saved);
            debugLog('Cart loaded from storage', cart.length);
        } catch (error) {
            debugError('Error loading cart', error);
            cart = [];
        }
    }
    updateCartIcon();
}

function getCartCount() {
    return cart.reduce((sum, item) => sum + item.quantity, 0);
}

function getCartTotal() {
    return cart.reduce((sum, item) => sum + ((item.sellPrice || item.sell_price || item.price || 0) * item.quantity), 0);
}

// ====================================================
// CART ICON - FIXED VERSION
// ====================================================

function updateCartIcon() {
    debugLog('Updating cart icon...');
    
    let cartIcon = document.getElementById('sales-cart-icon');

    if (!cartIcon) {
        debugLog('Creating new cart icon');
        cartIcon = document.createElement('div');
        cartIcon.id = 'sales-cart-icon';
        document.body.appendChild(cartIcon);
        
        // Add CSS styles for cart icon
        addCartIconStyles();
    }

    const count = getCartCount();
    const total = getCartTotal();

    debugLog(`Cart state - Count: ${count}, Total: $${total.toFixed(2)}`);

    cartIcon.innerHTML = `
        <div class="cart-icon-container" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 50px;
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            border: 2px solid white;
            display: flex;
            align-items: center;
            gap: 8px;
            user-select: none;
            transition: transform 0.2s, box-shadow 0.2s;
        ">
            🛒 ${count} items | $${total.toFixed(2)}
        </div>
    `;

    const container = cartIcon.querySelector('.cart-icon-container');
    
    // Add click handler
    container.onclick = () => {
        debugLog('Cart icon clicked!');
        if (count > 0) {
            showCartReview();
        } else {
            showNotification('Cart is empty! Add items first.', 'info', 2000);
        }
    };
    
    // Add hover effects
    container.onmouseenter = () => {
        container.style.transform = 'scale(1.05)';
        container.style.boxShadow = '0 6px 25px rgba(102, 126, 234, 0.6)';
    };
    
    container.onmouseleave = () => {
        container.style.transform = 'scale(1)';
        container.style.boxShadow = '0 4px 20px rgba(102, 126, 234, 0.4)';
    };
    
    // Add bounce animation when items added
    if (count > 0) {
        container.style.animation = 'cartBounce 0.4s ease';
        setTimeout(() => {
            container.style.animation = '';
        }, 400);
    }
    
    debugLog('Cart icon updated successfully');
}

function addCartIconStyles() {
    if (!document.getElementById('cart-icon-styles')) {
        const style = document.createElement('style');
        style.id = 'cart-icon-styles';
        style.textContent = `
            #sales-cart-icon {
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 9990;
                max-width: calc(100vw - 40px);
                overflow: hidden;
            }
            
            .cart-icon-container {
                position: relative;
                min-width: 180px;
                text-align: center;
            }
            
            @keyframes cartBounce {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            /* Cart Modal Styles */
            .cart-modal-backdrop {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                z-index: 9999;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                animation: fadeIn 0.3s ease;
            }
            
            .cart-modal-container {
                background: white;
                border-radius: 20px;
                width: 100%;
                max-width: 600px;
                max-height: 85vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                animation: slideUp 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes slideUp {
                from { transform: translateY(20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
        debugLog('Cart icon styles added');
    }
}

// ====================================================
// ADD ITEM TO CART (FIXED FOR SEPARATE BASE/SELLING UNITS)
// ====================================================

function addItemToCart(item) {
    console.log('🛒 Adding item to cart:', item);
    
    if (!item || !item.name) {
        console.error('🛒 Invalid item:', item);
        return false;
    }
    
    // Always add ONE item
    const qty = 1;
    const stock = item.stock || item.available_stock || item.batch_remaining || 0;
    
    // Check stock
    if (stock < qty) {
        showNotification(`❌ "${item.name}" is out of stock!`, 'error', 3000);
        return false;
    }
    
    // ✅ CRITICAL: Create UNIQUE cart ID
    // Base item: item_id + batch_id
    // Selling unit: item_id + sell_unit_id + batch_id
    const cartItemId = item.type === 'selling_unit' 
        ? `${item.item_id}_${item.sell_unit_id}_${item.batch_id}`
        : `${item.item_id}_main_${item.batch_id}`;
    
    const cartItem = {
        // ✅ UNIQUE ID for cart
        id: cartItemId,
        
        // Core IDs
        item_id: item.item_id || item.id || Date.now(),
        main_item_id: item.main_item_id || item.item_id || item.id || Date.now(),
        
        // Names
        name: item.name,
        display_name: item.display_name || item.name,
        
        // Quantity & Pricing
        quantity: qty,
        sellPrice: item.sellPrice || item.sell_price || item.price || 0,
        sell_price: item.sellPrice || item.sell_price || item.price || 0,
        price: item.price || item.sellPrice || item.sell_price || 0,
        
        // ✅ CRITICAL: CATEGORY ID FOR BACKEND
        category_id: item.category_id || 'unknown',
        category_name: item.category_name || 'Uncategorized',
        
        // Stock
        stock: stock,
        available_stock: stock,
        
        // ✅ CRITICAL: TYPE MUST BE PRESERVED
        type: item.type || 'main_item',
        
        // Batch Info
        batch_id: item.batch_id || item.batchId || null,
        batchId: item.batch_id || item.batchId || null,
        batch_name: item.batch_name || null,
        batch_remaining: item.batch_remaining || stock,
        
        // Selling Unit Info (only for selling units)
        sell_unit_id: item.sell_unit_id || null,
        conversion_factor: item.conversion_factor || 1,
        
        // Optional
        thumbnail: item.thumbnail || null,
        added_at: new Date().toISOString()
    };
    
    console.log('🛒 Processed cart item:', cartItem);
    console.log('🛒 Cart item ID:', cartItemId);
    console.log('🛒 Item type:', cartItem.type);
    
    // ✅ CRITICAL: Find existing item using UNIQUE ID (not item_id alone!)
    const existingIndex = cart.findIndex(i => i.id === cartItemId);
    
    if (existingIndex !== -1) {
        // Update existing item
        const newQuantity = cart[existingIndex].quantity + qty;
        if (stock < newQuantity) {
            showNotification(`❌ Cannot add more. Only ${stock - cart[existingIndex].quantity} available`, 'error', 3000);
            return false;
        }
        cart[existingIndex].quantity = newQuantity;
        console.log('🛒 Updated existing item:', cart[existingIndex].name, 'x', cart[existingIndex].quantity, 'Type:', cart[existingIndex].type);
    } else {
        // Add new item
        cart.push(cartItem);
        console.log('🛒 Added new item:', cartItem.name, 'Type:', cartItem.type, 'Batch:', cartItem.batch_id);
    }
    
    saveCartToStorage();
    updateCartIcon();
    
    const itemName = cartItem.display_name || cartItem.name;
    showNotification(`✅ Added ${itemName} to cart!`, 'success', 2000);
    
    return true;
}

// ====================================================
// NOTIFICATION SYSTEM
// ====================================================

function showNotification(message, type = 'info', duration = 3000) {
    // Remove existing notification
    const existing = document.getElementById('cart-notification');
    if (existing) existing.remove();
    
    const colors = {
        info: { bg: '#3498db', icon: 'ℹ️' },
        success: { bg: '#2ecc71', icon: '✅' },
        warning: { bg: '#f39c12', icon: '⚠️' },
        error: { bg: '#e74c3c', icon: '❌' }
    };
    
    const config = colors[type] || colors.info;
    
    const notification = document.createElement('div');
    notification.id = 'cart-notification';
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: ${config.bg};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10001;
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
    
    // Auto remove after duration
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
    
    return notification;
}

// ====================================================
// CART REVIEW MODAL (FIXED WITH TYPE INDICATORS)
// ====================================================

function showCartReview() {
    debugLog('Showing cart review');
    
    if (cart.length === 0) {
        showNotification('Cart is empty!', 'info', 2000);
        return;
    }
    
    // Remove existing modal
    const existingModal = document.querySelector('.cart-modal-backdrop');
    if (existingModal) existingModal.remove();

    const total = getCartTotal();
    
    // Create modal structure
    const modalBackdrop = document.createElement('div');
    modalBackdrop.className = 'cart-modal-backdrop';
    
    modalBackdrop.innerHTML = `
        <div class="cart-modal-container">
            <!-- Header -->
            <div style="
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h2 style="margin: 0; font-size: 24px; display: flex; align-items: center; gap: 10px;">
                    <span>🛒</span>
                    <span>Your Cart (${getCartCount()} items)</span>
                </h2>
                <button id="close-cart-btn" style="
                    background: rgba(255,255,255,0.2);
                    border: none;
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    font-size: 24px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: background 0.2s;
                ">×</button>
            </div>
            
            <!-- Items List -->
            <div style="
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                max-height: 50vh;
            ">
                ${cart.map((item, index) => {
                    const price = item.sellPrice || item.sell_price || item.price || 0;
                    const subtotal = price * item.quantity;
                    const itemName = item.display_name || item.name;
                    
                    // Type indicator
                    const typeBadge = item.type === 'selling_unit' 
                        ? `<span style="background:#9b59b6;color:white;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">Selling Unit</span>`
                        : `<span style="background:#3498db;color:white;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">Base Item</span>`;
                    
                    // Batch info
                    const batchInfo = item.batch_id ? `
                        <div style="
                            background: #e9ecef;
                            color: #7950f2;
                            font-size: 12px;
                            padding: 2px 8px;
                            border-radius: 4px;
                            display: inline-block;
                            margin-right: 8px;
                        ">Batch: ${item.batch_id}</div>
                    ` : '';
                    
                    // Selling unit conversion info
                    const conversionInfo = item.type === 'selling_unit' && item.conversion_factor 
                        ? `<div style="font-size:11px;color:#666;margin-top:2px;">1 Base = ${item.conversion_factor} ${item.display_name || 'units'}</div>`
                        : '';
                    
                    return `
                        <div class="cart-item" style="
                            padding: 16px;
                            margin-bottom: 12px;
                            background: #f8f9fa;
                            border-radius: 12px;
                            border: 1px solid #e9ecef;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            transition: transform 0.2s, box-shadow 0.2s;
                        ">
                            <div style="flex: 1;">
                                <div style="
                                    font-weight: 600;
                                    color: #333;
                                    font-size: 16px;
                                    margin-bottom: 4px;
                                    display: flex;
                                    align-items: center;
                                ">
                                    ${itemName} ${typeBadge}
                                </div>
                                <div style="
                                    color: #666;
                                    font-size: 14px;
                                    margin-bottom: 4px;
                                ">$${price.toFixed(2)} × ${item.quantity}</div>
                                ${batchInfo}
                                ${conversionInfo}
                            </div>
                            <div style="
                                display: flex;
                                align-items: center;
                                gap: 16px;
                            ">
                                <div style="
                                    font-weight: 700;
                                    color: #2ed573;
                                    font-size: 18px;
                                ">
                                    $${subtotal.toFixed(2)}
                                </div>
                                <button onclick="window.cartIcon.removeItem(${index})" style="
                                    background: #ff6b6b;
                                    color: white;
                                    border: none;
                                    width: 36px;
                                    height: 36px;
                                    border-radius: 8px;
                                    font-size: 20px;
                                    cursor: pointer;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    transition: background 0.2s;
                                ">×</button>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
            
            <!-- Footer -->
            <div style="
                padding: 24px;
                border-top: 2px solid #e9ecef;
                background: #f8f9fa;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                ">
                    <div>
                        <div style="font-size: 14px; color: #666; margin-bottom: 4px;">Total Amount</div>
                        <div style="font-size: 32px; font-weight: 800; color: #333;">$${total.toFixed(2)}</div>
                    </div>
                    <button id="clear-all-btn" style="
                        padding: 12px 24px;
                        background: #f8f9fa;
                        border: 2px solid #ff6b6b;
                        color: #ff6b6b;
                        border-radius: 10px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.2s;
                    ">Clear All</button>
                </div>
                
                <div style="display: flex; gap: 12px;">
                    <button id="continue-shopping-btn" style="
                        flex: 1;
                        padding: 16px;
                        background: #e9ecef;
                        color: #666;
                        border: none;
                        border-radius: 12px;
                        font-weight: 600;
                        font-size: 16px;
                        cursor: pointer;
                        transition: background 0.2s;
                    ">Continue Shopping</button>
                    <button id="checkout-btn" style="
                        flex: 1;
                        padding: 16px;
                        background: linear-gradient(135deg, #2ed573, #1dd1a1);
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-weight: 600;
                        font-size: 16px;
                        cursor: pointer;
                        transition: transform 0.2s, box-shadow 0.2s;
                    ">Proceed to Checkout →</button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalBackdrop);
    
    // Add hover effects for cart items
    setTimeout(() => {
        const items = document.querySelectorAll('.cart-item');
        items.forEach(item => {
            item.onmouseenter = () => {
                item.style.transform = 'translateY(-2px)';
                item.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
            };
            item.onmouseleave = () => {
                item.style.transform = 'translateY(0)';
                item.style.boxShadow = 'none';
            };
        });
    }, 100);
    
    // Add event handlers
    document.getElementById('close-cart-btn').onclick = () => {
        modalBackdrop.remove();
    };
    
    document.getElementById('clear-all-btn').onclick = () => {
        if (confirm('Are you sure you want to clear all items from your cart?')) {
            cart = [];
            saveCartToStorage();
            updateCartIcon();
            modalBackdrop.remove();
            showNotification('Cart cleared successfully!', 'success', 2000);
        }
    };
    
    document.getElementById('continue-shopping-btn').onclick = () => {
        modalBackdrop.remove();
    };
    
    document.getElementById('checkout-btn').onclick = () => {
        modalBackdrop.remove();
        setTimeout(() => showPaymentModal(), 300);
    };
    
    // Close modal when clicking outside
    modalBackdrop.onclick = (e) => {
        if (e.target === modalBackdrop) {
            modalBackdrop.remove();
        }
    };
    
    debugLog('Cart review modal shown');
}

// ====================================================
// PAYMENT MODAL (FIXED)
// ====================================================

function showPaymentModal() {
    debugLog('Showing payment modal');
    
    const total = getCartTotal();
    
    // Remove existing modal
    const existingModal = document.querySelector('.cart-modal-backdrop');
    if (existingModal) existingModal.remove();
    
    const modalBackdrop = document.createElement('div');
    modalBackdrop.className = 'cart-modal-backdrop';
    
    modalBackdrop.innerHTML = `
        <div class="cart-modal-container" style="max-width: 500px;">
            <!-- Header -->
            <div style="
                background: linear-gradient(135deg, #1dd1a1, #10ac84);
                color: white;
                padding: 24px;
                text-align: center;
            ">
                <h2 style="margin: 0; font-size: 24px; display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <span>💳</span>
                    <span>Complete Purchase</span>
                </h2>
                <div style="margin-top: 16px; font-size: 14px; opacity: 0.9;">
                    Complete your purchase by confirming payment
                </div>
            </div>
            
            <!-- Payment Details -->
            <div style="padding: 24px;">
                <div style="text-align: center; margin-bottom: 32px;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Total Amount</div>
                    <div style="font-size: 48px; font-weight: 800; color: #333; margin-bottom: 8px;">
                        $${total.toFixed(2)}
                    </div>
                    <div style="color: #666; font-size: 14px;">
                        ${cart.length} item${cart.length !== 1 ? 's' : ''} in cart
                    </div>
                </div>
                
                <!-- Payment Method -->
                <div style="margin-bottom: 24px;">
                    <div style="font-weight: 600; color: #333; margin-bottom: 12px;">Payment Method</div>
                    <div style="
                        background: #f8f9fa;
                        border: 2px solid #e9ecef;
                        border-radius: 12px;
                        padding: 16px;
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    ">
                        <div style="
                            width: 40px;
                            height: 40px;
                            background: #2ed573;
                            border-radius: 8px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: white;
                            font-size: 20px;
                        ">💰</div>
                        <div>
                            <div style="font-weight: 600; color: #333;">Cash</div>
                            <div style="color: #666; font-size: 14px;">Pay with cash</div>
                        </div>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div style="display: flex; gap: 12px;">
                    <button id="back-to-cart-btn" style="
                        flex: 1;
                        padding: 16px;
                        background: #e9ecef;
                        color: #666;
                        border: none;
                        border-radius: 12px;
                        font-weight: 600;
                        font-size: 16px;
                        cursor: pointer;
                        transition: background 0.2s;
                    ">← Back to Cart</button>
                    <button id="complete-purchase-btn" style="
                        flex: 1;
                        padding: 16px;
                        background: linear-gradient(135deg, #2ed573, #1dd1a1);
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-weight: 600;
                        font-size: 16px;
                        cursor: pointer;
                        transition: transform 0.2s, box-shadow 0.2s;
                    ">
                        <span style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                            <span>Complete Purchase</span>
                            <span>✅</span>
                        </span>
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modalBackdrop);
    
    // Add event handlers
    document.getElementById('back-to-cart-btn').onclick = () => {
        modalBackdrop.remove();
        setTimeout(() => showCartReview(), 300);
    };
    
    document.getElementById('complete-purchase-btn').onclick = async () => {
        const btn = document.getElementById('complete-purchase-btn');
        const originalText = btn.innerHTML;
        
        btn.innerHTML = '<span>Processing...</span>';
        btn.disabled = true;
        
        try {
            await completeSale({
                method: 'cash',
                cashAmount: total,
                notes: 'Sale from cart system'
            });
            
            // Success - close modal
            modalBackdrop.remove();
            
        } catch (error) {
            // Error - reset button and show error
            btn.innerHTML = originalText;
            btn.disabled = false;
            showNotification(`❌ Sale failed: ${error.message}`, 'error', 5000);
        }
    };
    
    // Close modal when clicking outside
    modalBackdrop.onclick = (e) => {
        if (e.target === modalBackdrop) {
            modalBackdrop.remove();
        }
    };
}

// ====================================================
// COMPLETE SALE FUNCTION (UPDATED)
// ====================================================

async function completeSale(paymentDetails = {}) {
    console.log('🛒 Starting sale completion');
    
    const auth = getAuth();
    const user = auth.currentUser;
    if (!user) {
        throw new Error("Please login first");
    }

    if (!currentShopId) {
        currentShopId = user.uid;
    }

    if (cart.length === 0) {
        throw new Error("Cart is empty!");
    }

    // Prepare sale data with ALL required fields
    const saleData = {
        shop_id: currentShopId,
        user_id: user.uid,
        seller: {
            type: localStorage.getItem("sessionType") || "owner",
            authUid: user.uid,
            name: user.displayName || "",
            email: user.email || ""
        },
        items: cart.map(item => ({
            // Core IDs
            item_id: item.item_id,
            main_item_id: item.main_item_id || item.item_id,
            
            // ✅ CRITICAL: MUST INCLUDE CATEGORY_ID
            category_id: item.category_id || 'unknown',
            
            // Item info
            name: item.name,
            display_name: item.display_name || item.name,
            
            // ✅ CRITICAL: TYPE MUST BE PRESERVED
            type: item.type || "main_item",
            
            // Quantity & Pricing
            quantity: item.quantity,
            price: item.price || item.sellPrice || item.sell_price || 0,
            sellPrice: item.price || item.sellPrice || item.sell_price || 0,
            
            // Batch info
            batch_id: item.batch_id || item.batchId || null,
            batchId: item.batch_id || item.batchId || null,
            batch_remaining: item.batch_remaining || 0,
            
            // ✅ CRITICAL: Selling unit info (only for selling units)
            sell_unit_id: item.sell_unit_id || null,
            conversion_factor: item.conversion_factor || 1,
            
            // Unit (for backend)
            unit: item.type === 'selling_unit' ? (item.display_name || 'unit') : 'unit'
        })),
        payment: paymentDetails,
        timestamp: new Date().toISOString()
    };

    console.log('🛒 Sending sale data to backend:', saleData);
    console.log('🛒 Items breakdown:', saleData.items.map(item => ({
        name: item.name,
        type: item.type,
        batch_id: item.batch_id,
        sell_unit_id: item.sell_unit_id,
        conversion_factor: item.conversion_factor
    })));

    try {
        const response = await fetch(`${FLASK_BACKEND_URL}/complete-sale`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(saleData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Sale failed: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log('🛒 Sale successful:', result);
        
        // Clear cart
        cart = [];
        saveCartToStorage();
        updateCartIcon();
        
        showNotification('✅ Sale completed successfully!', 'success', 5000);
        
        return result;
        
    } catch (error) {
        console.error('🛒 Sale error:', error);
        throw error;
    }
}

// ====================================================
// GLOBAL FUNCTIONS FOR CART ITEM REMOVAL
// ====================================================

function removeCartItem(index) {
    if (index >= 0 && index < cart.length) {
        const itemName = cart[index].name;
        cart.splice(index, 1);
        saveCartToStorage();
        updateCartIcon();
        showNotification(`Removed ${itemName} from cart`, 'info', 2000);
        
        // Refresh cart modal if open
        const existingModal = document.querySelector('.cart-modal-backdrop');
        if (existingModal) {
            existingModal.remove();
            if (cart.length > 0) {
                setTimeout(() => showCartReview(), 300);
            }
        }
    }
}

// ====================================================
// INITIALIZATION
// ====================================================

document.addEventListener("DOMContentLoaded", () => {
    debugLog('Cart system initializing...');
    
    // Load existing cart
    loadCartFromStorage();
    updateCartIcon();
    
    // Test if cart icon is working
    setTimeout(() => {
        const cartIcon = document.getElementById('sales-cart-icon');
        if (cartIcon) {
            console.log('✅ Cart system initialized');
            console.log(`✅ Cart items: ${cart.length}`);
            console.log(`✅ Cart total: $${getCartTotal().toFixed(2)}`);
        } else {
            console.error('❌ Cart icon not found - retrying...');
            updateCartIcon();
        }
    }, 100);
    
    // Expose globally
    window.cartIcon = {
        addItem: addItemToCart,
        getCart: () => [...cart],
        clearCart: () => {
            cart = [];
            saveCartToStorage();
            updateCartIcon();
            showNotification('Cart cleared', 'info', 2000);
        },
        removeItem: removeCartItem,
        showCart: showCartReview,
        updateIcon: updateCartIcon,
        getCount: getCartCount,
        getTotal: getCartTotal,
        debug: () => {
            console.log('🛒 CART DEBUG:', cart.map(item => ({
                name: item.name,
                type: item.type,
                id: item.id,
                batch_id: item.batch_id,
                sell_unit_id: item.sell_unit_id,
                quantity: item.quantity
            })));
        }
    };
    
    debugLog('Cart system ready!');
});

// Export main function
export { addItemToCart };