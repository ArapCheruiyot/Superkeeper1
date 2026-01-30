// cart-icon.js - SMART CART SYSTEM WITH BACKEND INTEGRATION (DRAGGABLE)

import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";
import { db } from "./firebase-config.js";
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

const FLASK_BACKEND_URL = window.location.origin;

// ====================================================
// GLOBAL CART STATE
// ====================================================
let cart = [];
let currentShopId = null;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let initialX = 0;
let initialY = 0;

// ====================================================
// DEBUG UTILITIES
// ====================================================

function debugLog(message, data = null) {
    console.log(`🛒 ${message}`, data || '');
}

// ====================================================
// CART MANAGEMENT FUNCTIONS
// ====================================================

function saveCartToStorage() {
    localStorage.setItem('smart_sales_cart', JSON.stringify(cart));
    debugLog('Cart saved to storage', cart.length);
}

function loadCartFromStorage() {
    const saved = localStorage.getItem('smart_sales_cart');
    if (saved) {
        try {
            cart = JSON.parse(saved);
            debugLog('Cart loaded from storage', cart.length);
        } catch (error) {
            console.error('Error loading cart', error);
            cart = [];
        }
    }
    updateCartIcon();
}

function getCartCount() {
    return cart.reduce((sum, item) => sum + item.quantity, 0);
}

function getCartTotal() {
    return cart.reduce((sum, item) => sum + ((item.price || item.sellPrice || item.sell_price || 0) * item.quantity), 0);
}

// ====================================================
// DRAGGABLE CART ICON
// ====================================================

function updateCartIcon() {
    debugLog('Updating cart icon...');
    
    let cartIcon = document.getElementById('sales-cart-icon');

    if (!cartIcon) {
        cartIcon = document.createElement('div');
        cartIcon.id = 'sales-cart-icon';
        document.body.appendChild(cartIcon);
        addCartIconStyles();
        
        // Load saved position
        const savedPos = localStorage.getItem('cart_icon_position');
        if (savedPos) {
            try {
                const { x, y } = JSON.parse(savedPos);
                cartIcon.style.left = `${x}px`;
                cartIcon.style.top = `${y}px`;
            } catch (e) {
                console.warn('Failed to load cart icon position:', e);
            }
        }
    }

    const count = getCartCount();
    const total = getCartTotal();

    cartIcon.innerHTML = `
        <div class="cart-icon-container" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 50px;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            border: 2px solid white;
            display: flex;
            align-items: center;
            gap: 6px;
            user-select: none;
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
        ">
            <span style="font-size: 16px;">🛒</span>
            <div>
                <div style="font-size: 12px; line-height: 1.2;">${count} items</div>
                <div style="font-size: 10px; opacity: 0.9;">$${total.toFixed(2)}</div>
            </div>
            <div class="drag-handle" style="
                position: absolute;
                top: 2px;
                right: 2px;
                width: 12px;
                height: 12px;
                background: rgba(255,255,255,0.3);
                border-radius: 50%;
                cursor: move;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 8px;
            ">✥</div>
        </div>
    `;

    const container = cartIcon.querySelector('.cart-icon-container');
    const dragHandle = cartIcon.querySelector('.drag-handle');
    
    // Click to open cart
    container.onclick = (e) => {
        if (isDragging) {
            isDragging = false;
            return;
        }
        if (count > 0) {
            showCartReview();
        } else {
            showNotification('Cart is empty! Add items first.', 'info', 2000);
        }
    };
    
    // Drag functionality
    dragHandle.onmousedown = startDrag;
    dragHandle.ontouchstart = startDragTouch;
    
    container.onmouseenter = () => {
        if (!isDragging) {
            container.style.transform = 'scale(1.05)';
            container.style.boxShadow = '0 6px 25px rgba(102, 126, 234, 0.6)';
        }
    };
    
    container.onmouseleave = () => {
        if (!isDragging) {
            container.style.transform = 'scale(1)';
            container.style.boxShadow = '0 4px 20px rgba(102, 126, 234, 0.4)';
        }
    };
    
    if (count > 0) {
        container.style.animation = 'cartBounce 0.4s ease';
        setTimeout(() => container.style.animation = '', 400);
    }
    
    debugLog('Cart icon updated');
}

function startDrag(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const cartIcon = document.getElementById('sales-cart-icon');
    if (!cartIcon) return;
    
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    initialX = cartIcon.offsetLeft;
    initialY = cartIcon.offsetTop;
    
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', stopDrag);
    
    cartIcon.querySelector('.cart-icon-container').style.cursor = 'grabbing';
    cartIcon.querySelector('.cart-icon-container').style.boxShadow = '0 8px 30px rgba(102, 126, 234, 0.8)';
}

function startDragTouch(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const cartIcon = document.getElementById('sales-cart-icon');
    if (!cartIcon) return;
    
    isDragging = true;
    dragStartX = e.touches[0].clientX;
    dragStartY = e.touches[0].clientY;
    initialX = cartIcon.offsetLeft;
    initialY = cartIcon.offsetTop;
    
    document.addEventListener('touchmove', dragTouch);
    document.addEventListener('touchend', stopDragTouch);
    
    cartIcon.querySelector('.cart-icon-container').style.cursor = 'grabbing';
    cartIcon.querySelector('.cart-icon-container').style.boxShadow = '0 8px 30px rgba(102, 126, 234, 0.8)';
}

function drag(e) {
    if (!isDragging) return;
    
    e.preventDefault();
    const cartIcon = document.getElementById('sales-cart-icon');
    if (!cartIcon) return;
    
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;
    
    let newX = initialX + dx;
    let newY = initialY + dy;
    
    // Keep within viewport bounds
    const maxX = window.innerWidth - cartIcon.offsetWidth;
    const maxY = window.innerHeight - cartIcon.offsetHeight;
    
    newX = Math.max(0, Math.min(newX, maxX));
    newY = Math.max(0, Math.min(newY, maxY));
    
    cartIcon.style.left = `${newX}px`;
    cartIcon.style.top = `${newY}px`;
}

function dragTouch(e) {
    if (!isDragging) return;
    
    e.preventDefault();
    const cartIcon = document.getElementById('sales-cart-icon');
    if (!cartIcon) return;
    
    const dx = e.touches[0].clientX - dragStartX;
    const dy = e.touches[0].clientY - dragStartY;
    
    let newX = initialX + dx;
    let newY = initialY + dy;
    
    // Keep within viewport bounds
    const maxX = window.innerWidth - cartIcon.offsetWidth;
    const maxY = window.innerHeight - cartIcon.offsetHeight;
    
    newX = Math.max(0, Math.min(newX, maxX));
    newY = Math.max(0, Math.min(newY, maxY));
    
    cartIcon.style.left = `${newX}px`;
    cartIcon.style.top = `${newY}px`;
}

function stopDrag() {
    if (!isDragging) return;
    
    isDragging = false;
    document.removeEventListener('mousemove', drag);
    document.removeEventListener('mouseup', stopDrag);
    
    const cartIcon = document.getElementById('sales-cart-icon');
    if (cartIcon) {
        cartIcon.querySelector('.cart-icon-container').style.cursor = 'pointer';
        cartIcon.querySelector('.cart-icon-container').style.boxShadow = '0 4px 20px rgba(102, 126, 234, 0.4)';
        
        // Save position
        saveCartPosition();
    }
}

function stopDragTouch() {
    if (!isDragging) return;
    
    isDragging = false;
    document.removeEventListener('touchmove', dragTouch);
    document.removeEventListener('touchend', stopDragTouch);
    
    const cartIcon = document.getElementById('sales-cart-icon');
    if (cartIcon) {
        cartIcon.querySelector('.cart-icon-container').style.cursor = 'pointer';
        cartIcon.querySelector('.cart-icon-container').style.boxShadow = '0 4px 20px rgba(102, 126, 234, 0.4)';
        
        // Save position
        saveCartPosition();
    }
}

function saveCartPosition() {
    const cartIcon = document.getElementById('sales-cart-icon');
    if (cartIcon) {
        const position = {
            x: cartIcon.offsetLeft,
            y: cartIcon.offsetTop
        };
        localStorage.setItem('cart_icon_position', JSON.stringify(position));
    }
}

function addCartIconStyles() {
    if (!document.getElementById('cart-icon-styles')) {
        const style = document.createElement('style');
        style.id = 'cart-icon-styles';
        style.textContent = `
            #sales-cart-icon {
                position: fixed;
                top: 40px;
                right: 50px;
                z-index: 9990;
                max-width: 160px;
                overflow: hidden;
                cursor: move;
                user-select: none;
            }
            
            .cart-icon-container {
                position: relative;
                min-width: 140px;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .cart-icon-container:hover .drag-handle {
                opacity: 1;
            }
            
            .drag-handle {
                opacity: 0.5;
                transition: opacity 0.2s;
            }
            
            .drag-handle:hover {
                opacity: 1;
                background: rgba(255,255,255,0.5);
            }
            
            @keyframes cartBounce {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
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
            
            @media (max-width: 480px) {
                #sales-cart-icon {
                    max-width: 140px;
                    right: 10px;
                }
                
                .cart-icon-container {
                    min-width: 120px;
                    font-size: 12px;
                    padding: 6px 12px;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// ====================================================
// ADD ITEM TO CART (UPDATED FOR SMART SYSTEM)
// ====================================================

function addItemToCart(item) {
    console.log('🛒 Adding smart item to cart:', item);
    
    if (!item || !item.name) {
        console.error('Invalid item:', item);
        return false;
    }
    
    const qty = 1; // One-tap system
    
    // ✅ Use smart fields from backend
    const stock = item.real_available !== undefined ? item.real_available : item.batch_remaining;
    
    if (stock < qty && item.can_fulfill === false) {
        showNotification(`❌ "${item.name}" is out of stock!`, 'error', 3000);
        return false;
    }
    
    // Create unique cart ID
    const cartItemId = item.type === 'selling_unit' 
        ? `${item.item_id}_${item.sell_unit_id}_${item.batch_id}`
        : `${item.item_id}_main_${item.batch_id}`;
    
    const cartItem = {
        // Core identification
        id: cartItemId,
        cart_item_id: cartItemId,
        item_id: item.item_id,
        main_item_id: item.main_item_id || item.item_id,
        
        // Item info
        name: item.name,
        display_name: item.display_name || item.name,
        
        // Quantity & pricing
        quantity: qty,
        price: item.price || item.sellPrice || item.sell_price || 0,
        sellPrice: item.price || item.sellPrice || item.sell_price || 0,
        sell_price: item.price || item.sellPrice || item.sell_price || 0,
        
        // Required for backend
        category_id: item.category_id || 'unknown',
        category_name: item.category_name || 'Uncategorized',
        
        // Stock info
        stock: stock,
        available_stock: stock,
        
        // Smart fields
        type: item.type || 'main_item',
        batch_id: item.batch_id,
        batchId: item.batch_id,
        batch_name: item.batch_name,
        batch_remaining: item.batch_remaining || stock,
        
        // Selling unit info
        sell_unit_id: item.sell_unit_id,
        conversion_factor: item.conversion_factor || 1,
        
        // Smart system fields
        can_fulfill: item.can_fulfill !== undefined ? item.can_fulfill : true,
        batch_switch_required: item.batch_switch_required || false,
        is_current_batch: item.is_current_batch || false,
        real_available: item.real_available,
        
        // Metadata
        thumbnail: item.thumbnail,
        added_at: new Date().toISOString(),
        _batch_switched: item._batch_switched || false
    };
    
    console.log('🛒 Smart cart item:', {
        id: cartItem.id,
        type: cartItem.type,
        batch_id: cartItem.batch_id,
        can_fulfill: cartItem.can_fulfill
    });
    
    // Find existing item by unique ID
    const existingIndex = cart.findIndex(i => i.id === cartItemId);
    
    if (existingIndex !== -1) {
        const newQuantity = cart[existingIndex].quantity + qty;
        if (stock < newQuantity) {
            showNotification(`❌ Only ${stock - cart[existingIndex].quantity} available`, 'error', 3000);
            return false;
        }
        cart[existingIndex].quantity = newQuantity;
        console.log('🛒 Updated existing item:', cart[existingIndex].name, 'x', cart[existingIndex].quantity);
    } else {
        cart.push(cartItem);
        console.log('🛒 Added new item:', cartItem.name, 'Type:', cartItem.type);
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
    
    return notification;
}

// ====================================================
// CART REVIEW MODAL (ENHANCED)
// ====================================================

function showCartReview() {
    debugLog('Showing smart cart review');
    
    if (cart.length === 0) {
        showNotification('Cart is empty!', 'info', 2000);
        return;
    }
    
    const existingModal = document.querySelector('.cart-modal-backdrop');
    if (existingModal) existingModal.remove();

    const total = getCartTotal();
    
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
                    const price = item.price || item.sellPrice || item.sell_price || 0;
                    const subtotal = price * item.quantity;
                    const itemName = item.display_name || item.name;
                    
                    // Type indicator
                    const typeBadge = item.type === 'selling_unit' 
                        ? `<span style="background:#9b59b6;color:white;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">Selling Unit</span>`
                        : `<span style="background:#3498db;color:white;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">Base Item</span>`;
                    
                    // Batch info
                    const batchInfo = item.batch_name ? `
                        <div style="
                            background: #e9ecef;
                            color: #7950f2;
                            font-size: 12px;
                            padding: 2px 8px;
                            border-radius: 4px;
                            display: inline-block;
                            margin-right: 8px;
                        ">${item.batch_name}</div>
                    ` : '';
                    
                    // Smart indicator
                    const smartIndicator = item._batch_switched ? `
                        <div style="
                            background: #ff9f43;
                            color: white;
                            font-size: 10px;
                            padding: 2px 6px;
                            border-radius: 4px;
                            display: inline-block;
                            margin-right: 8px;
                        ">Auto-switched</div>
                    ` : '';
                    
                    // Stock info
                    const stockInfo = item.real_available !== undefined ? `
                        <div style="font-size:11px;color:#666;margin-top:2px;">
                            Real stock: ${item.real_available.toFixed(2)}
                        </div>
                    ` : '';
                    
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
                                ${smartIndicator}
                                ${stockInfo}
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
    
    // Event handlers
    document.getElementById('close-cart-btn').onclick = () => modalBackdrop.remove();
    
    document.getElementById('clear-all-btn').onclick = () => {
        if (confirm('Clear all items from cart?')) {
            cart = [];
            saveCartToStorage();
            updateCartIcon();
            modalBackdrop.remove();
            showNotification('Cart cleared!', 'success', 2000);
        }
    };
    
    document.getElementById('continue-shopping-btn').onclick = () => modalBackdrop.remove();
    
    document.getElementById('checkout-btn').onclick = () => {
        modalBackdrop.remove();
        setTimeout(() => showPaymentModal(), 300);
    };
    
    modalBackdrop.onclick = (e) => {
        if (e.target === modalBackdrop) modalBackdrop.remove();
    };
}

// ====================================================
// PAYMENT MODAL
// ====================================================

function showPaymentModal() {
    debugLog('Showing payment modal');
    
    const total = getCartTotal();
    
    const existingModal = document.querySelector('.cart-modal-backdrop');
    if (existingModal) existingModal.remove();
    
    const modalBackdrop = document.createElement('div');
    modalBackdrop.className = 'cart-modal-backdrop';
    
    modalBackdrop.innerHTML = `
        <div class="cart-modal-container" style="max-width: 500px;">
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
                    Smart batch system ensures correct stock allocation
                </div>
            </div>
            
            <div style="padding: 24px;">
                <div style="text-align: center; margin-bottom: 32px;">
                    <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Total Amount</div>
                    <div style="font-size: 48px; font-weight: 800; color: #333; margin-bottom: 8px;">
                        $${total.toFixed(2)}
                    </div>
                    <div style="color: #666; font-size: 14px;">
                        ${cart.length} item${cart.length !== 1 ? 's' : ''} • Smart batch tracking
                    </div>
                </div>
                
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
                notes: 'Sale from smart batch system'
            });
            
            modalBackdrop.remove();
            
        } catch (error) {
            btn.innerHTML = originalText;
            btn.disabled = false;
            showNotification(`❌ Sale failed: ${error.message}`, 'error', 5000);
        }
    };
    
    modalBackdrop.onclick = (e) => {
        if (e.target === modalBackdrop) modalBackdrop.remove();
    };
}

// ====================================================
// COMPLETE SALE FUNCTION (UPDATED)
// ====================================================

async function completeSale(paymentDetails = {}) {
    console.log('🛒 Starting smart sale completion');
    
    const auth = getAuth();
    const user = auth.currentUser;
    if (!user) throw new Error("Please login first");

    if (!currentShopId) currentShopId = user.uid;
    if (cart.length === 0) throw new Error("Cart is empty!");

    // Prepare smart sale data
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
            category_id: item.category_id || 'unknown',
            
            // Item info
            name: item.name,
            display_name: item.display_name || item.name,
            type: item.type || "main_item",
            
            // Quantity & Pricing
            quantity: item.quantity,
            price: item.price || item.sellPrice || item.sell_price || 0,
            sellPrice: item.price || item.sellPrice || item.sell_price || 0,
            
            // Batch info
            batch_id: item.batch_id,
            batchId: item.batch_id,
            batch_remaining: item.batch_remaining || 0,
            
            // Smart fields
            can_fulfill: item.can_fulfill !== undefined ? item.can_fulfill : true,
            batch_switch_required: item.batch_switch_required || false,
            real_available: item.real_available,
            
            // Selling unit info
            sell_unit_id: item.sell_unit_id,
            conversion_factor: item.conversion_factor || 1,
            
            // Unit for backend
            unit: item.type === 'selling_unit' ? (item.display_name || 'unit') : 'unit'
        })),
        payment: paymentDetails,
        timestamp: new Date().toISOString(),
        cart_id: localStorage.getItem('current_cart_id') // Track which cart
    };

    console.log('🛒 Sending smart sale data:', saleData);

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
        console.log('🛒 Smart sale successful:', result);
        
        // Clear cart and reset cart ID
        cart = [];
        localStorage.removeItem('current_cart_id');
        saveCartToStorage();
        updateCartIcon();
        
        showNotification('✅ Smart sale completed successfully!', 'success', 5000);
        
        return result;
        
    } catch (error) {
        console.error('🛒 Sale error:', error);
        throw error;
    }
}

// ====================================================
// CART ITEM REMOVAL
// ====================================================

function removeCartItem(index) {
    if (index >= 0 && index < cart.length) {
        const itemName = cart[index].name;
        cart.splice(index, 1);
        saveCartToStorage();
        updateCartIcon();
        showNotification(`Removed ${itemName} from cart`, 'info', 2000);
        
        const existingModal = document.querySelector('.cart-modal-backdrop');
        if (existingModal) {
            existingModal.remove();
            if (cart.length > 0) setTimeout(() => showCartReview(), 300);
        }
    }
}

// ====================================================
// INITIALIZATION
// ====================================================

document.addEventListener("DOMContentLoaded", () => {
    console.log('🛒 Smart Cart System Initializing...');
    
    loadCartFromStorage();
    updateCartIcon();
    
    // Test cart icon
    setTimeout(() => {
        if (!document.getElementById('sales-cart-icon')) {
            console.error('Cart icon not found - retrying...');
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
            console.log('🛒 SMART CART DEBUG:', cart.map(item => ({
                name: item.name,
                type: item.type,
                id: item.id,
                batch_id: item.batch_id,
                can_fulfill: item.can_fulfill,
                quantity: item.quantity
            })));
        }
    };
    
    console.log('🛒 Smart Cart System Ready! (Draggable)');
    console.log(`
╔═══════════════════════════════════════════╗
║     🧠 SMART CART SYSTEM READY           ║
╠═══════════════════════════════════════════╣
║ • Smart batch tracking                   ║
║ • Real stock management                  ║
║ • Cart-aware functionality               ║
║ • DRAGGABLE CART ICON                    ║
║ • Integrated with sales.js               ║
╚═══════════════════════════════════════════╝
`);
});

// Export main function
export { addItemToCart };