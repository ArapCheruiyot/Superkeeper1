// businessIntelligence.js - COMPLETE SELF-CONTAINED BUSINESS INTELLIGENCE MODULE
import { db } from "./firebase-config.js";
import { 
    collection, 
    query, 
    onSnapshot, 
    orderBy, 
    limit,
    getDocs,
    doc,
    getDoc
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

document.addEventListener("DOMContentLoaded", () => {
    console.log("🚀 Business Intelligence module loaded");
    
    let businessIntelligenceOverlay = null;
    let currentShopId = null;
    let realTimeListeners = [];
    
    const NAV_HEIGHT = 64;
    
    // ===========================================
    // 1. INJECT CSS STYLES (UPDATED WITH CART FIX)
    // ===========================================
    function injectStyles() {
        if (document.getElementById('business-intelligence-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'business-intelligence-styles';
        style.textContent = `
            /* Business Intelligence Overlay */
            #business-intelligence-overlay {
                position: fixed;
                top: ${NAV_HEIGHT}px;
                left: 0;
                width: 100%;
                height: calc(100vh - ${NAV_HEIGHT}px);
                background: #f8f9fa;
                z-index: 2000;
                display: none;
                flex-direction: column;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                overflow: hidden;
            }
            
            /* Purple Action Button - FIXED POSITION */
            #business-intelligence-btn {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 14px 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                flex: 1;
                min-width: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                margin: 10px 5px; /* Added margin for spacing */
            }
            
            #business-intelligence-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            /* Adjust action buttons container to prevent cart overlap */
            .action-buttons {
                display: flex;
                flex-direction: column;
                gap: 15px;
                margin-bottom: 100px !important; /* Push content above cart */
                padding-bottom: 20px;
            }
            
            /* Mobile responsive layout */
            @media (max-width: 768px) {
                .action-buttons {
                    flex-direction: column;
                }
                
                #business-intelligence-btn {
                    min-width: 100%;
                    order: 3; /* Make it appear last */
                }
            }
            
            /* Cart icon adjustment when BI button is present */
            body.has-bi-button #sales-cart-icon {
                bottom: 100px !important; /* Move cart up to avoid overlap */
            }
            
            /* BI Header */
            .bi-header {
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                flex-shrink: 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            /* Time Period Buttons */
            .time-period-btn {
                padding: 8px 16px;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                color: white;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .time-period-btn:hover {
                background: rgba(255,255,255,0.2);
            }
            
            .time-period-btn.active {
                background: rgba(255,255,255,0.3);
                font-weight: 600;
            }
            
            /* Metrics Cards */
            .metric-card {
                background: white;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 24px rgba(0,0,0,0.1);
            }
            
            .metric-value {
                font-size: 36px;
                font-weight: 800;
                margin: 10px 0;
                line-height: 1;
            }
            
            /* Chart Containers */
            .chart-container {
                background: white;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #e9ecef;
                margin-bottom: 20px;
            }
            
            /* Insight Cards */
            .insight-card {
                background: white;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            
            .insight-card.warning {
                border-left-color: #ff6b6b;
                background: #fff5f5;
            }
            
            .insight-card.success {
                border-left-color: #2ed573;
                background: #f0fff4;
            }
            
            /* Loading Animation */
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .loading-pulse {
                animation: pulse 1.5s infinite;
            }
            
            /* Responsive Grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            /* Tablet and Desktop */
            @media (min-width: 768px) {
                .metrics-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
                
                .action-buttons {
                    flex-direction: row !important;
                    flex-wrap: wrap;
                }
                
                #business-intelligence-btn {
                    min-width: calc(50% - 20px);
                }
            }
            
            /* Desktop */
            @media (min-width: 1024px) {
                .metrics-grid {
                    grid-template-columns: repeat(4, 1fr);
                }
                
                .action-buttons {
                    flex-direction: row;
                }
                
                #business-intelligence-btn {
                    min-width: 200px;
                    flex: 1;
                }
            }
            
            /* Scrollbar Styling */
            #business-intelligence-overlay ::-webkit-scrollbar {
                width: 8px;
            }
            
            #business-intelligence-overlay ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            
            #business-intelligence-overlay ::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 4px;
            }
            
            #business-intelligence-overlay ::-webkit-scrollbar-thumb:hover {
                background: #a1a1a1;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    // ===========================================
    // 2. INJECT ACTION BUTTON (UPDATED)
    // ===========================================
    function injectActionButton() {
        // Check if button already exists
        if (document.getElementById('business-intelligence-btn')) return;
        
        // Find action buttons container
        const actionButtons = document.querySelector('.action-buttons');
        if (!actionButtons) {
            console.error('Action buttons container not found');
            return;
        }
        
        // Add class to body for CSS targeting
        document.body.classList.add('has-bi-button');
        
        // Create the new button
        const biButton = document.createElement('button');
        biButton.id = 'business-intelligence-btn';
        biButton.innerHTML = `
            <span style="display: flex; align-items: center; gap: 8px;">
                📊 My Business Intelligence
            </span>
        `;
        
        // Add click event
        biButton.addEventListener('click', openBusinessIntelligenceOverlay);
        
        // Append to action buttons
        actionButtons.appendChild(biButton);
        
        // Adjust cart icon position if it exists
        adjustCartIconPosition();
        
        console.log('✅ Business Intelligence button injected');
    }
    
    function adjustCartIconPosition() {
        const cartIcon = document.getElementById('sales-cart-icon');
        if (cartIcon) {
            cartIcon.style.bottom = '100px'; // Move cart icon up
        }
    }
    
    // ===========================================
    // 3. CREATE OVERLAY
    // ===========================================
    function createBusinessIntelligenceOverlay() {
        if (businessIntelligenceOverlay) return;
        
        businessIntelligenceOverlay = document.createElement("div");
        businessIntelligenceOverlay.id = "business-intelligence-overlay";
        document.body.appendChild(businessIntelligenceOverlay);
    }
    
    // ===========================================
    // 4. LOAD BUSINESS INTELLIGENCE DATA (FIXED - MOVE THIS UP)
    // ===========================================
    async function loadBusinessIntelligence(timePeriod = 'today') {
        console.log(`📊 Loading business intelligence for: ${timePeriod}`);
        
        const content = document.getElementById('bi-content');
        if (!content) return;
        
        try {
            const auth = getAuth();
            const user = auth.currentUser;
            if (!user) throw new Error('User not logged in');
            
            // Get shop ID
            if (!currentShopId) {
                let shopId = user.uid;
                const snap = await getDoc(doc(db, "Users", shopId));
                if (snap.exists() && snap.data().shop_id) {
                    shopId = snap.data().shop_id;
                }
                currentShopId = shopId;
            }
            
            // Calculate date range based on time period
            const dateRange = getDateRange(timePeriod);
            
            // Load all data in parallel
            const [salesData, inventoryData] = await Promise.all([
                getSalesData(dateRange),
                getInventoryData()
            ]);
            
            // Calculate insights
            const insights = calculateBusinessInsights(salesData, inventoryData, dateRange);
            
            // Render the dashboard
            renderBusinessIntelligence(insights, timePeriod);
            
        } catch (error) {
            console.error('Error loading business intelligence:', error);
            const content = document.getElementById('bi-content');
            if (content) {
                content.innerHTML = `
                    <div style="text-align: center; padding: 60px 20px;">
                        <div style="font-size: 48px; margin-bottom: 20px;">😕</div>
                        <h3 style="margin: 0 0 8px; color: #ff6b6b;">Failed to load data</h3>
                        <p style="margin: 0; font-size: 14px; color: #888;">${error.message}</p>
                        <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">
                            Retry
                        </button>
                    </div>
                `;
            }
        }
    }
    
    // ===========================================
    // 5. DATA FETCHING FUNCTIONS
    // ===========================================
    function getDateRange(period) {
        const now = new Date();
        const start = new Date();
        
        switch(period) {
            case 'today':
                start.setHours(0, 0, 0, 0);
                break;
            case 'week':
                start.setDate(now.getDate() - 7);
                break;
            case 'month':
                start.setMonth(now.getMonth() - 1);
                break;
            case 'all':
                start.setFullYear(now.getFullYear() - 1);
                break;
            default:
                start.setHours(0, 0, 0, 0);
        }
        
        return { start, end: now };
    }
    
    async function getSalesData(dateRange) {
        try {
            if (!currentShopId) return { total: 0, items: [], transactions: [] };
            
            const receiptsRef = collection(db, "Shops", currentShopId, "receipts");
            const receiptsSnapshot = await getDocs(receiptsRef);
            
            const salesData = {
                total: 0,
                items: [],
                transactions: []
            };
            
            receiptsSnapshot.forEach(doc => {
                const data = doc.data();
                const receiptDate = data.timestamp?.toDate?.() || new Date(data.timestamp || data.created_at);
                
                // Filter by date range
                if (receiptDate >= dateRange.start && receiptDate <= dateRange.end) {
                    salesData.total += data.total || 0;
                    salesData.transactions.push({
                        id: data.receipt_id,
                        total: data.total,
                        date: receiptDate,
                        items: data.items || []
                    });
                    
                    // Aggregate items
                    if (data.items) {
                        data.items.forEach(item => {
                            const existing = salesData.items.find(i => i.id === item.id);
                            if (existing) {
                                existing.quantity += item.quantity;
                                existing.revenue += (item.sellPrice || 0) * item.quantity;
                            } else {
                                salesData.items.push({
                                    id: item.id,
                                    name: item.name,
                                    quantity: item.quantity,
                                    revenue: (item.sellPrice || 0) * item.quantity,
                                    price: item.sellPrice || 0
                                });
                            }
                        });
                    }
                }
            });
            
            return salesData;
        } catch (error) {
            console.error('Error fetching sales data:', error);
            return { total: 0, items: [], transactions: [] };
        }
    }
    
    async function getInventoryData() {
        try {
            if (!currentShopId) return { items: [], totalValue: 0, lowStock: [] };
            
            const categoriesRef = collection(db, "Shops", currentShopId, "categories");
            const categoriesSnapshot = await getDocs(categoriesRef);
            
            const inventoryData = {
                items: [],
                totalValue: 0,
                lowStock: [],
                outOfStock: []
            };
            
            for (const categoryDoc of categoriesSnapshot.docs) {
                const itemsRef = collection(categoryDoc.ref, "items");
                const itemsSnapshot = await getDocs(itemsRef);
                
                itemsSnapshot.forEach(itemDoc => {
                    const data = itemDoc.data();
                    const stock = data.stock || 0;
                    const sellPrice = data.sellPrice || 0;
                    const buyPrice = data.buyPrice || 0;
                    const value = stock * buyPrice;
                    
                    const item = {
                        id: itemDoc.id,
                        name: data.name || 'Unnamed Item',
                        category: categoryDoc.data().name || 'Uncategorized',
                        stock: stock,
                        sellPrice: sellPrice,
                        buyPrice: buyPrice,
                        stockValue: value,
                        images: data.images || [],
                        lowStockAlert: data.lowStockAlert || 5,
                        lastUpdated: data.updatedAt || data.createdAt
                    };
                    
                    inventoryData.items.push(item);
                    inventoryData.totalValue += value;
                    
                    // Check stock status
                    if (stock === 0) {
                        inventoryData.outOfStock.push(item);
                    } else if (stock <= item.lowStockAlert) {
                        inventoryData.lowStock.push(item);
                    }
                });
            }
            
            return inventoryData;
        } catch (error) {
            console.error('Error fetching inventory data:', error);
            return { items: [], totalValue: 0, lowStock: [], outOfStock: [] };
        }
    }
    
    // ===========================================
    // 6. CALCULATE INSIGHTS
    // ===========================================
    function calculateBusinessInsights(salesData, inventoryData, dateRange) {
        const insights = {
            totalRevenue: salesData.total || 0,
            averageSale: salesData.transactions.length > 0 
                ? salesData.total / salesData.transactions.length 
                : 0,
            transactionsCount: salesData.transactions.length || 0,
            
            totalStockValue: inventoryData.totalValue || 0,
            totalItems: inventoryData.items.length || 0,
            lowStockCount: inventoryData.lowStock.length || 0,
            outOfStockCount: inventoryData.outOfStock.length || 0,
            
            topSellingItems: salesData.items
                .sort((a, b) => b.revenue - a.revenue)
                .slice(0, 5),
            
            lowStockItems: inventoryData.lowStock
                .sort((a, b) => a.stock - b.stock)
                .slice(0, 10),
            
            outOfStockItems: inventoryData.outOfStock,
            
            estimatedProfit: salesData.total - (inventoryData.totalValue * 0.7),
            
            dailySales: calculateDailySales(salesData.transactions, dateRange),
            
            actionableInsights: generateActionableInsights(salesData, inventoryData)
        };
        
        return insights;
    }
    
    function calculateDailySales(transactions, dateRange) {
        const daily = {};
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        
        transactions.forEach(txn => {
            const day = days[txn.date.getDay()];
            daily[day] = (daily[day] || 0) + txn.total;
        });
        
        return Object.entries(daily).map(([day, total]) => ({ day, total }));
    }
    
    function generateActionableInsights(salesData, inventoryData) {
        const insights = [];
        
        // Low stock insights
        if (inventoryData.lowStock.length > 0) {
            const lowStockNames = inventoryData.lowStock
                .slice(0, 3)
                .map(item => item.name)
                .join(', ');
            
            insights.push({
                type: 'warning',
                title: '⚠️ Low Stock Alert',
                message: `${inventoryData.lowStock.length} items are running low: ${lowStockNames}`,
                action: 'Restock soon to avoid lost sales'
            });
        }
        
        // Out of stock insights
        if (inventoryData.outOfStock.length > 0) {
            insights.push({
                type: 'critical',
                title: '🚨 Out of Stock',
                message: `${inventoryData.outOfStock.length} items are completely out of stock`,
                action: 'Reorder immediately'
            });
        }
        
        // Top sellers insights
        if (salesData.items.length > 0) {
            const topSeller = salesData.items[0];
            if (topSeller) {
                insights.push({
                    type: 'success',
                    title: '🏆 Top Performer',
                    message: `${topSeller.name} generated $${topSeller.revenue.toFixed(2)} in revenue`,
                    action: 'Consider stocking more or creating promotions'
                });
            }
        }
        
        // Stock value insight
        if (inventoryData.totalValue > 0) {
            insights.push({
                type: 'info',
                title: '💰 Inventory Value',
                message: `Your current inventory is worth $${inventoryData.totalValue.toFixed(2)}`,
                action: 'Review stock levels to optimize cash flow'
            });
        }
        
        return insights;
    }
    
    // ===========================================
    // 7. RENDER DASHBOARD
    // ===========================================
    function renderBusinessIntelligence(insights, timePeriod) {
        const content = document.getElementById('bi-content');
        if (!content) return;
        
        const periodText = {
            'today': 'Today',
            'week': 'This Week',
            'month': 'This Month',
            'all': 'All Time'
        }[timePeriod] || timePeriod;
        
        content.innerHTML = `
            <!-- Key Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Total Revenue</div>
                    <div class="metric-value" style="color: #2ed573;">$${insights.totalRevenue.toFixed(2)}</div>
                    <div style="font-size: 13px; color: #888;">${periodText}</div>
                </div>
                
                <div class="metric-card">
                    <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Stock Value</div>
                    <div class="metric-value" style="color: #667eea;">$${insights.totalStockValue.toFixed(2)}</div>
                    <div style="font-size: 13px; color: #888;">${insights.totalItems} items</div>
                </div>
                
                <div class="metric-card">
                    <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Transactions</div>
                    <div class="metric-value" style="color: #ff6b6b;">${insights.transactionsCount}</div>
                    <div style="font-size: 13px; color: #888;">Avg: $${insights.averageSale.toFixed(2)}</div>
                </div>
                
                <div class="metric-card">
                    <div style="font-size: 14px; color: #666; margin-bottom: 8px;">Stock Alerts</div>
                    <div class="metric-value" style="color: #ffa502;">
                        ${insights.lowStockCount + insights.outOfStockCount}
                    </div>
                    <div style="font-size: 13px; color: #888;">
                        ${insights.lowStockCount} low, ${insights.outOfStockCount} out
                    </div>
                </div>
            </div>
            
            <!-- Actionable Insights -->
            <div style="margin-bottom: 30px;">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">💡 Actionable Insights</h3>
                <div id="insights-list">
                    ${insights.actionableInsights.map(insight => `
                        <div class="insight-card ${insight.type === 'warning' ? 'warning' : insight.type === 'success' ? 'success' : ''}">
                            <div style="font-weight: 600; color: #333; margin-bottom: 4px;">
                                ${insight.title}
                            </div>
                            <div style="color: #666; font-size: 14px; margin-bottom: 6px;">
                                ${insight.message}
                            </div>
                            <div style="font-size: 13px; color: #667eea; font-weight: 500;">
                                💡 ${insight.action}
                            </div>
                        </div>
                    `).join('')}
                    
                    ${insights.actionableInsights.length === 0 ? `
                        <div class="insight-card success">
                            <div style="font-weight: 600; color: #333; margin-bottom: 4px;">
                                ✅ Everything Looks Good
                            </div>
                            <div style="color: #666; font-size: 14px;">
                                Your business is running smoothly. Keep up the good work!
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
            
            <!-- Top Selling Items -->
            <div class="chart-container" style="margin-bottom: 20px;">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">🏆 Top Selling Items (${periodText})</h3>
                <div style="max-height: 300px; overflow-y: auto;">
                    ${insights.topSellingItems.length > 0 ? insights.topSellingItems.map((item, index) => `
                        <div style="
                            display: flex;
                            align-items: center;
                            padding: 12px;
                            margin-bottom: 8px;
                            background: ${index < 3 ? '#f8f9fa' : 'white'};
                            border-radius: 10px;
                            border-left: 4px solid ${index === 0 ? '#ffd700' : index === 1 ? '#c0c0c0' : index === 2 ? '#cd7f32' : '#e9ecef'};
                        ">
                            <div style="width: 30px; text-align: center; font-weight: 800; color: #666;">
                                ${index + 1}
                            </div>
                            <div style="flex: 1;">
                                <div style="font-weight: 600; color: #333;">${item.name}</div>
                                <div style="font-size: 12px; color: #666;">
                                    ${item.quantity} units sold
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-weight: 700; color: #2ed573; font-size: 16px;">
                                    $${item.revenue.toFixed(2)}
                                </div>
                                <div style="font-size: 12px; color: #888;">
                                    $${item.price} each
                                </div>
                            </div>
                        </div>
                    `).join('') : `
                        <div style="text-align: center; padding: 40px 20px; color: #888; font-style: italic;">
                            No sales data for ${periodText.toLowerCase()}
                        </div>
                    `}
                </div>
            </div>
            
            <!-- Low Stock Items -->
            <div class="chart-container">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">📦 Items Needing Attention</h3>
                <div style="max-height: 300px; overflow-y: auto;">
                    ${insights.lowStockItems.length > 0 ? insights.lowStockItems.map(item => `
                        <div style="
                            display: flex;
                            align-items: center;
                            padding: 12px;
                            margin-bottom: 8px;
                            background: ${item.stock === 0 ? '#fff5f5' : '#fff5e6'};
                            border-radius: 10px;
                            border-left: 4px solid ${item.stock === 0 ? '#ff6b6b' : '#ffa502'};
                        ">
                            <div style="flex: 1;">
                                <div style="font-weight: 600; color: #333; margin-bottom: 4px;">
                                    ${item.name}
                                    ${item.stock === 0 ? '<span style="color: #ff6b6b; font-size: 12px; margin-left: 8px;">(OUT OF STOCK)</span>' : ''}
                                </div>
                                <div style="font-size: 12px; color: #666;">
                                    ${item.category} • Alert at: ${item.lowStockAlert} units
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 24px; font-weight: 800; color: ${item.stock === 0 ? '#ff6b6b' : '#ffa502'};">
                                    ${item.stock}
                                </div>
                                <div style="font-size: 12px; color: #888;">
                                    units remaining
                                </div>
                            </div>
                        </div>
                    `).join('') : `
                        <div style="text-align: center; padding: 40px 20px; color: #2ed573; font-weight: 500;">
                            ✅ All items have sufficient stock!
                        </div>
                    `}
                </div>
            </div>
            
            <!-- Daily Sales Chart -->
            ${insights.dailySales.length > 0 ? `
                <div class="chart-container">
                    <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">📈 Sales by Day</h3>
                    <div style="display: flex; align-items: flex-end; height: 200px; gap: 10px; padding: 20px 0;">
                        ${insights.dailySales.map(day => {
                            const maxValue = Math.max(...insights.dailySales.map(d => d.total));
                            const height = maxValue > 0 ? (day.total / maxValue) * 150 : 10;
                            return `
                                <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                                    <div style="
                                        width: 40px;
                                        height: ${height}px;
                                        background: linear-gradient(to top, #667eea, #764ba2);
                                        border-radius: 8px 8px 0 0;
                                        margin-bottom: 8px;
                                    "></div>
                                    <div style="font-size: 12px; color: #666; font-weight: 500;">
                                        ${day.day}
                                    </div>
                                    <div style="font-size: 11px; color: #888; margin-top: 2px;">
                                        $${day.total.toFixed(2)}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }
    
    // ===========================================
    // 8. OPEN/CLOSE OVERLAY (FIXED)
    // ===========================================
    async function openBusinessIntelligenceOverlay() {
        console.log('📊 Opening Business Intelligence...');
        
        // Ensure styles are injected
        injectStyles();
        
        // Create overlay if it doesn't exist
        createBusinessIntelligenceOverlay();
        
        // Show loading state
        businessIntelligenceOverlay.innerHTML = `
            <div class="bi-header">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div>
                        <h1 style="margin: 0; font-size: 24px; font-weight: 700;">📊 My Business Intelligence</h1>
                        <p style="margin: 6px 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                            Real-time insights for smarter decisions
                        </p>
                    </div>
                    <button id="close-business-intelligence" style="
                        background: rgba(255,255,255,0.2);
                        border: none;
                        color: white;
                        width: 44px;
                        height: 44px;
                        border-radius: 12px;
                        font-size: 22px;
                        cursor: pointer;
                        flex-shrink: 0;
                    ">×</button>
                </div>
                
                <!-- Time Period Selector -->
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <button class="time-period-btn active" data-period="today">Today</button>
                    <button class="time-period-btn" data-period="week">This Week</button>
                    <button class="time-period-btn" data-period="month">This Month</button>
                    <button class="time-period-btn" data-period="all">All Time</button>
                </div>
            </div>
            
            <div id="bi-content" style="flex: 1; overflow-y: auto; padding: 20px;">
                <div style="text-align: center; padding: 60px 20px;">
                    <div style="font-size: 48px; margin-bottom: 20px; opacity: 0.5; animation: pulse 1.5s infinite;">📊</div>
                    <h3 style="margin: 0 0 8px; color: #555; font-size: 18px;">Loading Business Intelligence...</h3>
                    <p style="margin: 0; font-size: 14px; color: #888;">Analyzing your data in real-time</p>
                </div>
            </div>
            
            <div style="padding: 15px; background: white; border-top: 1px solid #e9ecef; text-align: center; flex-shrink: 0;">
                <button id="refresh-bi" style="
                    padding: 12px 24px;
                    background: linear-gradient(135deg, #2ed573, #1dd1a1);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 14px;
                    cursor: pointer;
                ">
                    🔄 Refresh Insights
                </button>
            </div>
        `;
        
        // Show overlay
        businessIntelligenceOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        
        // Add event listeners
        document.getElementById('close-business-intelligence').onclick = closeBusinessIntelligenceOverlay;
        document.getElementById('refresh-bi').onclick = () => loadBusinessIntelligence('today');
        
        // Time period buttons
        businessIntelligenceOverlay.querySelectorAll('.time-period-btn').forEach(btn => {
            btn.onclick = (e) => {
                businessIntelligenceOverlay.querySelectorAll('.time-period-btn').forEach(b => {
                    b.classList.remove('active');
                });
                e.target.classList.add('active');
                loadBusinessIntelligence(e.target.dataset.period);
            };
        });
        
        // Load initial data
        await loadBusinessIntelligence('today');
    }
    
    function closeBusinessIntelligenceOverlay() {
        if (businessIntelligenceOverlay) {
            businessIntelligenceOverlay.style.display = 'none';
            document.body.style.overflow = '';
        }
    }
    
    // ===========================================
    // 9. REAL-TIME LISTENERS (OPTIONAL)
    // ===========================================
    function setupRealTimeListeners() {
        if (!currentShopId) return;
        
        try {
            // Listen for new receipts (sales)
            const receiptsRef = collection(db, "Shops", currentShopId, "receipts");
            const receiptsQuery = query(receiptsRef, orderBy("timestamp", "desc"), limit(5));
            
            const receiptsListener = onSnapshot(receiptsQuery, (snapshot) => {
                console.log('🔄 New sale detected, refreshing insights...');
                loadBusinessIntelligence('today');
            });
            
            realTimeListeners.push(receiptsListener);
            
            console.log('✅ Real-time listeners setup complete');
            
        } catch (error) {
            console.error('Error setting up real-time listeners:', error);
        }
    }
    
    function cleanUpRealTimeListeners() {
        realTimeListeners.forEach(unsubscribe => {
            if (typeof unsubscribe === 'function') {
                unsubscribe();
            }
        });
        realTimeListeners = [];
    }
    
    // ===========================================
    // 10. INITIALIZATION
    // ===========================================
    function initialize() {
        // Inject CSS
        injectStyles();
        
        // Inject button
        injectActionButton();
        
        console.log('✅ Business Intelligence module initialized');
    }
    
    // Start initialization
    initialize();
    
    // ===========================================
    // 11. EXPORT TO WINDOW
    // ===========================================
    window.openBusinessIntelligenceOverlay = openBusinessIntelligenceOverlay;
    window.closeBusinessIntelligenceOverlay = closeBusinessIntelligenceOverlay;
});