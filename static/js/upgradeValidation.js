// admin/modules/upgrade-requests.js
class UpgradeRequestsModule {
    constructor() {
        this.requests = [];
        this.filteredRequests = [];
        this.currentFilter = 'all';
        this.init();
    }

    async init() {
        console.log("📊 Upgrade Requests Module Initialized");
        // Module is ready, render will be called when needed
    }

    async render() {
        const container = document.getElementById('module-container');
        
        container.innerHTML = `
            <div class="module-header">
                <h2 class="module-title">
                    <i class="fas fa-rocket"></i>
                    Upgrade Requests
                </h2>
                <div class="header-actions">
                    <button class="btn btn-primary refresh-btn">
                        <i class="fas fa-sync-alt"></i>
                        Refresh
                    </button>
                </div>
            </div>
            
            <div class="module-content">
                <!-- Filters -->
                <div class="filters-bar">
                    <input type="text" class="search-box" placeholder="Search by shop name, reference..." id="searchRequests">
                    <select class="filter-select" id="statusFilter">
                        <option value="all">All Status</option>
                        <option value="pending_payment">Pending Payment</option>
                        <option value="payment_submitted">Payment Submitted</option>
                        <option value="pending_verification">Needs Verification</option>
                        <option value="completed">Completed</option>
                        <option value="cancelled">Cancelled</option>
                    </select>
                    <select class="filter-select" id="planFilter">
                        <option value="all">All Plans</option>
                        <option value="BASIC">Basic</option>
                        <option value="TEAM">Team</option>
                        <option value="BUSINESS">Business</option>
                        <option value="ENTERPRISE">Enterprise</option>
                    </select>
                </div>
                
                <!-- Stats Summary -->
                <div class="stats-summary" id="requestsStats">
                    <!-- Stats will be loaded here -->
                </div>
                
                <!-- Requests Table -->
                <div class="table-container">
                    <table class="data-table" id="requestsTable">
                        <thead>
                            <tr>
                                <th>Shop</th>
                                <th>Plan</th>
                                <th>Amount</th>
                                <th>Reference</th>
                                <th>Status</th>
                                <th>Requested</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="requestsTableBody">
                            <!-- Requests will be loaded here -->
                            <tr>
                                <td colspan="7" class="text-center">
                                    <div class="loading-state">
                                        <div class="spinner"></div>
                                        <p>Loading upgrade requests...</p>
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <!-- Pagination -->
                <div class="pagination" id="requestsPagination">
                    <!-- Pagination will be added here -->
                </div>
            </div>
        `;
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load data
        await this.loadRequests();
    }

    async loadRequests() {
        try {
            // Show loading
            document.getElementById('requestsTableBody').innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">
                        <div class="loading-state">
                            <div class="spinner"></div>
                            <p>Loading upgrade requests...</p>
                        </div>
                    </td>
                </tr>
            `;
            
            // Load requests from Firebase
            this.requests = await this.fetchRequestsFromFirebase();
            this.filteredRequests = [...this.requests];
            
            // Update UI
            this.updateStats();
            this.renderTable();
            this.renderPagination();
            
        } catch (error) {
            console.error("❌ Error loading requests:", error);
            document.getElementById('requestsTableBody').innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">
                        <div class="empty-state">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Error loading requests. Please try again.</p>
                        </div>
                    </td>
                </tr>
            `;
        }
    }

    async fetchRequestsFromFirebase() {
        // This is a simplified version - you'll need to implement the actual Firebase query
        const requests = [];
        
        // Example structure - you'll need to adjust based on your actual Firebase structure
        const shopsSnapshot = await window.db.collection("Shops").get();
        
        for (const shopDoc of shopsSnapshot.docs) {
            const shopId = shopDoc.id;
            const shopData = shopDoc.data();
            
            const requestsSnapshot = await window.db
                .collection("Shops")
                .doc(shopId)
                .collection("upgradeRequests")
                .orderBy("requestedAt", "desc")
                .get();
            
            requestsSnapshot.forEach(requestDoc => {
                const requestData = requestDoc.data();
                requests.push({
                    id: requestDoc.id,
                    shopId: shopId,
                    shopName: shopData.shopName || 'Unknown Shop',
                    shopEmail: shopData.email || '',
                    ...requestData
                });
            });
        }
        
        return requests;
    }

    setupEventListeners() {
        // Search
        document.getElementById('searchRequests').addEventListener('input', (e) => {
            this.filterRequests();
        });
        
        // Status filter
        document.getElementById('statusFilter').addEventListener('change', (e) => {
            this.filterRequests();
        });
        
        // Plan filter
        document.getElementById('planFilter').addEventListener('change', (e) => {
            this.filterRequests();
        });
        
        // Refresh button
        document.querySelector('.refresh-btn').addEventListener('click', () => {
            this.loadRequests();
        });
    }

    filterRequests() {
        const searchTerm = document.getElementById('searchRequests').value.toLowerCase();
        const statusFilter = document.getElementById('statusFilter').value;
        const planFilter = document.getElementById('planFilter').value;
        
        this.filteredRequests = this.requests.filter(request => {
            // Search filter
            const matchesSearch = !searchTerm || 
                request.shopName.toLowerCase().includes(searchTerm) ||
                request.shopEmail?.toLowerCase().includes(searchTerm) ||
                request.mpesaReference?.toLowerCase().includes(searchTerm);
            
            // Status filter
            const matchesStatus = statusFilter === 'all' || request.status === statusFilter;
            
            // Plan filter
            const matchesPlan = planFilter === 'all' || request.requestedPlan === planFilter;
            
            return matchesSearch && matchesStatus && matchesPlan;
        });
        
        this.renderTable();
        this.renderPagination();
    }

    updateStats() {
        const stats = {
            total: this.requests.length,
            pending: this.requests.filter(r => r.status === 'pending_payment').length,
            submitted: this.requests.filter(r => r.status === 'payment_submitted' || r.status === 'pending_verification').length,
            completed: this.requests.filter(r => r.status === 'completed').length,
            cancelled: this.requests.filter(r => r.status === 'cancelled').length
        };
        
        document.getElementById('requestsStats').innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-header">
                        <div class="stat-title">Total Requests</div>
                        <div class="stat-icon" style="background: rgba(102, 126, 234, 0.1); color: #667eea;">
                            <i class="fas fa-list"></i>
                        </div>
                    </div>
                    <div class="stat-value">${stats.total}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <div class="stat-title">Pending Payment</div>
                        <div class="stat-icon" style="background: rgba(245, 158, 11, 0.1); color: #f59e0b;">
                            <i class="fas fa-clock"></i>
                        </div>
                    </div>
                    <div class="stat-value">${stats.pending}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <div class="stat-title">Needs Verification</div>
                        <div class="stat-icon" style="background: rgba(59, 130, 246, 0.1); color: #3b82f6;">
                            <i class="fas fa-check-circle"></i>
                        </div>
                    </div>
                    <div class="stat-value">${stats.submitted}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-header">
                        <div class="stat-title">Completed</div>
                        <div class="stat-icon" style="background: rgba(16, 185, 129, 0.1); color: #10b981;">
                            <i class="fas fa-check"></i>
                        </div>
                    </div>
                    <div class="stat-value">${stats.completed}</div>
                </div>
            </div>
        `;
    }

    renderTable() {
        const tbody = document.getElementById('requestsTableBody');
        
        if (this.filteredRequests.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">
                        <div class="empty-state">
                            <i class="fas fa-search"></i>
                            <p>No upgrade requests match your filters.</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }
        
        // For now, just show first 20
        const requestsToShow = this.filteredRequests.slice(0, 20);
        
        tbody.innerHTML = requestsToShow.map(request => {
            const plan = window.PLANS?.[request.requestedPlan] || { name: request.requestedPlan, color: '#666' };
            const timeAgo = this.getTimeAgo(request.requestedAt?.toDate());
            
            let statusBadge = '';
            switch (request.status) {
                case 'pending_payment':
                    statusBadge = `<span class="badge badge-warning">Pending Payment</span>`;
                    break;
                case 'payment_submitted':
                    statusBadge = `<span class="badge badge-info">Payment Submitted</span>`;
                    break;
                case 'pending_verification':
                    statusBadge = `<span class="badge badge-info">Needs Verification</span>`;
                    break;
                case 'completed':
                    statusBadge = `<span class="badge badge-success">Completed</span>`;
                    break;
                case 'cancelled':
                    statusBadge = `<span class="badge badge-danger">Cancelled</span>`;
                    break;
                default:
                    statusBadge = `<span class="badge badge-secondary">${request.status}</span>`;
            }
            
            return `
                <tr>
                    <td>
                        <div class="shop-info">
                            <div class="user-avatar">${request.shopName.charAt(0)}</div>
                            <div>
                                <div class="shop-name">${request.shopName}</div>
                                <div class="shop-email">${request.shopEmail || ''}</div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <div class="plan-info">
                            <div class="plan-icon" style="background: ${plan.color}20; color: ${plan.color};">
                                ${plan.icon || '📊'}
                            </div>
                            <div>${plan.name}</div>
                        </div>
                    </td>
                    <td>
                        <div class="amount">KSh ${request.priceKES || 0}</div>
                    </td>
                    <td>
                        <div class="reference">
                            ${request.mpesaReference ? 
                                `<code>${request.mpesaReference}</code>` : 
                                `<span class="text-muted">No reference</span>`
                            }
                        </div>
                    </td>
                    <td>${statusBadge}</td>
                    <td>
                        <div class="time-ago">${timeAgo}</div>
                        <div class="date">${new Date(request.requestedAt?.toDate()).toLocaleDateString()}</div>
                    </td>
                    <td>
                        <div class="actions">
                            ${request.status === 'payment_submitted' || request.status === 'pending_verification' ? `
                                <button class="btn btn-success btn-sm" onclick="UpgradeRequestsModule.verifyRequest('${request.id}', '${request.shopId}')">
                                    <i class="fas fa-check"></i>
                                    Verify
                                </button>
                            ` : ''}
                            <button class="btn btn-secondary btn-sm" onclick="UpgradeRequestsModule.viewDetails('${request.id}', '${request.shopId}')">
                                <i class="fas fa-eye"></i>
                                View
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }

    renderPagination() {
        // Implement pagination if needed
        document.getElementById('requestsPagination').innerHTML = `
            <div class="pagination-info">
                Showing ${Math.min(this.filteredRequests.length, 20)} of ${this.filteredRequests.length} requests
            </div>
        `;
    }

    getTimeAgo(date) {
        if (!date) return 'Recently';
        const seconds = Math.floor((new Date() - date) / 1000);
        
        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    static async verifyRequest(requestId, shopId) {
        // Open verification modal
        console.log("Verify request:", requestId, shopId);
        // You can implement the verification logic here
        alert(`Verify request ${requestId} for shop ${shopId}`);
    }

    static async viewDetails(requestId, shopId) {
        console.log("View details:", requestId, shopId);
        // Open details modal
        alert(`View details for request ${requestId}`);
    }
}

// Export module
window.UpgradeRequestsModule = new UpgradeRequestsModule();