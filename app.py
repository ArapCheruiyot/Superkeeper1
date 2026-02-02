from flask import Flask, render_template, request, jsonify
import requests
import firebase_admin
from firebase_admin import credentials, firestore

import numpy as np

from io import BytesIO

# TODO: Re-enable TensorFlow imports when needed aslo this one from PIL import Image
# import tensorflow_hub as hub
# from embeddings import generate_embedding
# from sklearn.metrics.pairwise import cosine_similarity

import time
import base64
import math
import random 
import uuid
import json
from datetime import datetime, timedelta

# ======================================================
# APP INIT
# ======================================================
app = Flask(__name__)


# ======================================================
# FIREBASE CONFIG
# ======================================================
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    firebase_key = os.environ.get("FIREBASE_KEY")

    if not firebase_key:
        raise RuntimeError("FIREBASE_KEY environment variable not set")

    cred = credentials.Certificate(json.loads(firebase_key))
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ======================================================
# LOAD MODEL
# ======================================================
"""
print("[INIT] Loading TensorFlow Hub model...")
model = hub.load(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
)
print("[READY] Model loaded successfully.")
"""

# ======================================================
# FULL SHOP CACHE (STRICTLY PER SHOP) - UPDATED WITH BATCH TRACKING
# ======================================================
embedding_cache_full = {
    "shops": [],
    "last_updated": None,
    "total_shops": 0
}

def refresh_full_item_cache():
    """REVISED: Includes ALL items with BATCH tracking and selling units with batch links"""
    start = time.time()
    print("\n[INFO] Refreshing FULL shop cache (with batch tracking)...")

    shops_result = []

    for shop_doc in db.collection("Shops").stream():
        shop_id = shop_doc.id
        shop_data = shop_doc.to_dict()

        shop_entry = {
            "shop_id": shop_id,
            "shop_name": shop_data.get("name", ""),
            "categories": []
        }

        for cat_doc in shop_doc.reference.collection("categories").stream():
            cat_data = cat_doc.to_dict()
            cat_id = cat_doc.id

            category_entry = {
                "category_id": cat_id,
                "category_name": cat_data.get("name", ""),
                "items": []
            }

            for item_doc in cat_doc.reference.collection("items").stream():
                item_data = item_doc.to_dict()
                item_id = item_doc.id
                item_name = item_data.get("name", "Unnamed")

                # Get embeddings (if any)
                embeddings = []
                for emb_doc in item_doc.reference.collection("embeddings").stream():
                    vector = emb_doc.to_dict().get("vector")
                    if vector:
                        embeddings.append(np.array(vector))

                # Get batches for this item (NEW: batch breakdown)
                batches = item_data.get("batches", [])
                processed_batches = []
                for batch in batches:
                    processed_batches.append({
                        "batch_id": batch.get("id", f"batch_{int(time.time()*1000)}"),
                        "batch_name": batch.get("batchName", batch.get("batch_name", "Batch")),
                        "quantity": float(batch.get("quantity", 0)),
                        "remaining_quantity": float(batch.get("quantity", 0)),  # Will be updated during sales
                        "unit": batch.get("unit", "unit"),
                        "buy_price": float(batch.get("buyPrice", 0) or batch.get("buy_price", 0)),
                        "sell_price": float(batch.get("sellPrice", 0) or batch.get("sell_price", 0)),
                        "timestamp": batch.get("timestamp", 0),
                        "date": batch.get("date", ""),
                        "added_by": batch.get("addedBy", ""),
                        "selling_unit_allocations": batch.get("sellingUnitAllocations", {})  # Track allocations
                    })

                # Get selling units for this item with batch links (NEW)
                selling_units = []
                try:
                    # CORRECT PATH: Shops/{shop_id}/categories/{cat_id}/items/{item_id}/sellUnits
                    sell_units_ref = db.collection("Shops").document(shop_id) \
                        .collection("categories").document(cat_id) \
                        .collection("items").document(item_id) \
                        .collection("sellUnits")
                    
                    print(f"\n🔍 Checking selling units for item: {item_name}")
                    print(f"   Item ID: {item_id}")
                    print(f"   Category ID: {cat_id}")
                    print(f"   Collection path: Shops/{shop_id}/categories/{cat_id}/items/{item_id}/sellUnits")
                    
                    sell_units_docs = list(sell_units_ref.stream())
                    
                    print(f"   Found {len(sell_units_docs)} selling units")
                    
                    for sell_unit_doc in sell_units_docs:
                        sell_unit_data = sell_unit_doc.to_dict()
                        sell_unit_id = sell_unit_doc.id
                        
                        print(f"   Selling Unit: {sell_unit_data.get('name', 'No name')}")
                        print(f"     ID: {sell_unit_id}")
                        print(f"     Conversion Factor: {sell_unit_data.get('conversionFactor', 'Not set')}")
                        print(f"     Sell Price: {sell_unit_data.get('sellPrice', 'Not set')}")
                        
                        # Get batch links from selling unit (NEW)
                        batch_links = sell_unit_data.get("batchLinks", [])
                        total_units_available = 0
                        
                        # Calculate total available units from batch links
                        for link in batch_links:
                            total_units_available += link.get("maxUnitsAvailable", 0) - link.get("allocatedUnits", 0)
                        
                        selling_units.append({
                            "sell_unit_id": sell_unit_doc.id,
                            "name": sell_unit_data.get("name", ""),
                            "conversion_factor": float(sell_unit_data.get("conversionFactor", 1.0)),
                            "sell_price": float(sell_unit_data.get("sellPrice", 0.0)),
                            "images": sell_unit_data.get("images", []),
                            "is_base_unit": sell_unit_data.get("isBaseUnit", False),
                            "thumbnail": sell_unit_data.get("images", [None])[0] if sell_unit_data.get("images") else None,
                            "created_at": sell_unit_data.get("createdAt"),
                            "updated_at": sell_unit_data.get("updatedAt"),
                            # NEW: Batch tracking for selling units
                            "batch_links": batch_links,
                            "total_units_available": total_units_available,
                            "has_batch_links": len(batch_links) > 0
                        })
                    
                except Exception as e:
                    print(f"❌ ERROR fetching selling units: {e}")
                    # Don't crash, just continue

                # Calculate total stock from batches
                total_stock_from_batches = sum(batch.get("quantity", 0) for batch in batches)
                main_stock = float(item_data.get("stock", 0) or 0)
                
                # Use batch total if available, otherwise use main stock
                effective_stock = total_stock_from_batches if total_stock_from_batches > 0 else main_stock
                
                category_entry["items"].append({
                    "item_id": item_doc.id,
                    "name": item_data.get("name", ""),
                    "thumbnail": item_data.get("images", [None])[0],
                    "sell_price": float(item_data.get("sellPrice", 0) or 0),
                    "buy_price": float(item_data.get("buyPrice", 0) or 0),
                    "stock": effective_stock,
                    "base_unit": item_data.get("baseUnit", "unit"),
                    "embeddings": embeddings,
                    "has_embeddings": len(embeddings) > 0,
                    "selling_units": selling_units,
                    "category_id": category_entry["category_id"],
                    "category_name": category_entry["category_name"],
                    # NEW: Batch tracking
                    "batches": processed_batches,
                    "has_batches": len(processed_batches) > 0,
                    "total_stock_from_batches": total_stock_from_batches
                })

            # Only skip categories that have no items at all
            if category_entry["items"]:
                shop_entry["categories"].append(category_entry)

        # Only skip shops with no categories
        if shop_entry["categories"]:
            shops_result.append(shop_entry)

    embedding_cache_full["shops"] = shops_result
    embedding_cache_full["total_shops"] = len(shops_result)
    embedding_cache_full["last_updated"] = time.time()

    # Cache statistics
    total_main_items = 0
    total_selling_units = 0
    total_batches = 0
    for shop in shops_result:
        for category in shop["categories"]:
            total_main_items += len(category["items"])
            for item in category["items"]:
                total_selling_units += len(item.get("selling_units", []))
                total_batches += len(item.get("batches", []))

    print(f"\n[READY] Cached {len(shops_result)} shops, {total_main_items} main items, {total_selling_units} selling units, {total_batches} batches")
    print(f"[TIME] Cache refresh took {round((time.time()-start)*1000,2)}ms")
    
    return shops_result


def on_full_item_snapshot(col_snapshot, changes, read_time):
    """Listener for changes to main items"""
    print("[LISTENER] Main items changed → refreshing FULL cache")
    refresh_full_item_cache()


def on_selling_units_snapshot(col_snapshot, changes, read_time):
    """Listener for changes to selling units"""
    print("[LISTENER] Selling units changed → refreshing FULL cache")
    refresh_full_item_cache()


# ======================================================
# NEW: BATCH-AWARE FIFO HELPER FUNCTIONS
# ======================================================

def find_item_in_cache(shop_id, item_id):
    """Find item in cache by shop_id and item_id"""
    for shop in embedding_cache_full["shops"]:
        if shop["shop_id"] == shop_id:
            for category in shop["categories"]:
                for item in category["items"]:
                    if item["item_id"] == item_id:
                        return item
    return None

def find_selling_unit_in_cache(shop_id, item_id, sell_unit_id):
    """Find selling unit in cache"""
    item = find_item_in_cache(shop_id, item_id)
    if item:
        for sell_unit in item.get("selling_units", []):
            if sell_unit.get("sell_unit_id") == sell_unit_id:
                return sell_unit
    return None

def allocate_main_item_fifo(batches, requested_quantity):
    """
    Allocate quantity from batches using FIFO for main items
    Returns: {
        "success": True/False,
        "allocation": [{"batch_id": "...", "quantity": x, "price": y}, ...],
        "total_price": z
    }
    """
    if not batches:
        return {"success": False, "error": "No batches available"}
    
    # Sort batches by timestamp (oldest first)
    sorted_batches = sorted(batches, key=lambda x: x.get("timestamp", 0))
    
    allocation = []
    remaining = requested_quantity
    total_price = 0
    
    for batch in sorted_batches:
        if remaining <= 0:
            break
        
        available = batch.get("remaining_quantity", 0)
        if available > 0:
            take = min(available, remaining)
            batch_price = batch.get("sell_price", 0)
            
            allocation.append({
                "batch_id": batch["batch_id"],
                "batch_name": batch.get("batch_name", "Batch"),
                "quantity": take,
                "price": batch_price,
                "unit": batch.get("unit", "unit"),
                "batch_info": batch
            })
            
            total_price += take * batch_price
            remaining -= take
    
    if remaining > 0:
        return {"success": False, "error": f"Insufficient stock. Only {requested_quantity - remaining} available"}
    
    return {"success": True, "allocation": allocation, "total_price": total_price}

def allocate_selling_unit_fifo(batch_links, requested_units, conversion_factor):
    """
    Allocate selling units from batch links using FIFO
    Returns allocation in MAIN units for stock deduction
    """
    if not batch_links:
        return {"success": False, "error": "No batch links available"}
    
    # Sort batch links (FIFO - we need to get batch timestamps from cache)
    # For now, use the order they appear (should be FIFO if created properly)
    sorted_links = sorted(batch_links, key=lambda x: x.get("batchTimestamp", 0))
    
    allocation = []
    remaining_units = requested_units
    total_price = 0
    
    for link in sorted_links:
        if remaining_units <= 0:
            break
        
        available_units = link.get("maxUnitsAvailable", 0) - link.get("allocatedUnits", 0)
        if available_units > 0:
            take_units = min(available_units, remaining_units)
            price_per_unit = link.get("pricePerUnit", 0)
            
            # Convert to main units for stock deduction
            take_main_units = take_units / conversion_factor
            
            allocation.append({
                "batch_id": link.get("batchId"),
                "units_taken": take_units,
                "main_units_taken": take_main_units,
                "price_per_unit": price_per_unit,
                "total_for_batch": take_units * price_per_unit
            })
            
            total_price += take_units * price_per_unit
            remaining_units -= take_units
    
    if remaining_units > 0:
        return {"success": False, "error": f"Insufficient units. Only {requested_units - remaining_units} available"}
    
    return {"success": True, "allocation": allocation, "total_price": total_price}

# PLANS
PLANS_CONFIG = {
    "SOLO": {
        "id": "SOLO",
        "name": "Solo",
        "staff_limit": 0,
        "price_kes": 0,
        "description": "Perfect for individual entrepreneurs",
        "features": [
            {"text": "1 seat only (owner)", "included": True},
            {"text": "Up to 50 items", "included": True},
            {"text": "Basic stock tracking", "included": True},
            {"text": "Mobile app access", "included": True},
            {"text": "No concurrent staff access", "included": False},
            {"text": "No priority support", "included": False}
        ],
        "button_text": "Start Free Forever",
        "button_class": "btn-free",
        "best_for": "Perfect for individual entrepreneurs"
    },
    "BASIC": {
        "id": "BASIC",
        "name": "Basic",
        "staff_limit": 5,
        "price_kes": 250,
        "description": "Small business with employees",
        "features": [
            {"text": "Up to 5 concurrent seats", "included": True},
            {"text": "Up to 200 items", "included": True},
            {"text": "Basic staff access", "included": True},
            {"text": "Stock alerts", "included": True},
            {"text": "WhatsApp support", "included": True},
            {"text": "Data backup", "included": True}
        ],
        "button_text": "Pay via M-Pesa",
        "button_class": "btn-primary",
        "best_for": "Best for: Family shops & startups"
    },
    "TEAM": {
        "id": "TEAM",
        "name": "Team",
        "staff_limit": 10,
        "price_kes": 500,
        "description": "Growing business with team",
        "features": [
            {"text": "3-5 concurrent seats", "included": True},
            {"text": "Up to 500 items", "included": True},
            {"text": "Multiple staff roles (RBAC)", "included": True},
            {"text": "Sales reports & analytics", "included": True},
            {"text": "Data export (CSV/Excel)", "included": True},
            {"text": "Priority WhatsApp support", "included": True}
        ],
        "button_text": "Pay via M-Pesa",
        "button_class": "btn-primary btn-featured",
        "best_for": "Best value for growing businesses",
        "featured": True
    },
    "BUSINESS": {
        "id": "BUSINESS",
        "name": "Business",
        "staff_limit": 20,
        "price_kes": 1000,
        "description": "Multiple counters/locations",
        "features": [
            {"text": "6-10 concurrent seats", "included": True},
            {"text": "Unlimited items", "included": True},
            {"text": "Advanced analytics dashboard", "included": True},
            {"text": "Multi-location support", "included": True},
            {"text": "Custom categories", "included": True},
            {"text": "24/7 phone support", "included": True}
        ],
        "button_text": "Pay via M-Pesa",
        "button_class": "btn-primary",
        "best_for": "For established businesses"
    },
    "ENTERPRISE": {
        "id": "ENTERPRISE",
        "name": "Enterprise",
        "staff_limit": 50,
        "price_kes": 3000,
        "description": "Supermarkets & large operations",
        "features": [
            {"text": "11-20+ concurrent seats", "included": True},
            {"text": "Unlimited everything", "included": True},
            {"text": "API access", "included": True},
            {"text": "Dedicated account manager", "included": True},
            {"text": "Custom feature requests", "included": True},
            {"text": "On-site training available", "included": True}
        ],
        "button_text": "Contact Us",
        "button_class": "btn-enterprise",
        "best_for": "Custom solutions available"
    }
}

# ======================================================
# ROUTES
# ======================================================
# Option 3: Simple individual routes (recommended for clarity)
@app.route("/")
def home():
    return render_template(
        "home.html",
        title="Superkeeper - Inventory POS for Small Businesses",
        meta_desc="Mobile-first POS and inventory for small businesses. Start free, upgrade as you grow.",
        active_page="home"
    )

@app.route("/features")
def features():
    return render_template(
        "features.html",
        title="Features - Superkeeper",
        meta_desc="Everything you need, nothing you don't. Mobile-first POS, staff control, alerts, and more.",
        active_page="features"
    )

@app.route("/pricing")
def pricing():
    # Calculate annual discounts
    annual_discounts = []
    for plan_id, plan in PLANS_CONFIG.items():
        if plan["price_kes"] > 0 and plan_id != "ENTERPRISE":
            annual_price = plan["price_kes"] * 12
            discounted_price = int(annual_price * 0.8)
            savings = annual_price - discounted_price
            
            annual_discounts.append({
                "plan_name": plan["name"],
                "old_price": annual_price,
                "new_price": discounted_price,
                "savings": savings
            })
    
    return render_template(
        "pricing.html",
        title="Pricing - Superkeeper",
        meta_desc="Simple, seat-based pricing. Start free, upgrade as you grow.",
        active_page="pricing",
        plans=PLANS_CONFIG.values(),  # Pass all plans to template
        annual_discounts=annual_discounts,
        featured_plan="TEAM"
    )

@app.route("/testimonials")
def testimonials():
    return render_template(
        "testimonials.html",
        title="Success Stories - Superkeeper",
        meta_desc="Real results from real shops. See how Superkeeper helps small businesses.",
        active_page="testimonials"
    )

@app.route("/story")
def story():
    return render_template(
        "story.html",
        title="Our Story - Superkeeper",
        meta_desc="How Superkeeper was built for small businesses with big dreams.",
        active_page="story"
    )

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ======================================================
# VECTORIZE ITEM (STOCK IMAGE → EMBEDDING)
# ======================================================
@app.route("/vectorize-item", methods=["POST"])
def vectorize_item():
    try:
        data = request.get_json(force=True)

        required = [
            "event",
            "image_url",
            "item_id",
            "shop_id",
            "category_id",
            "image_index",
            "timestamp",
        ]

        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"status": "error", "missing_fields": missing}), 400

        print(f"📥 /vectorize-item → {data['item_id']} image {data['image_index']}")

        response = requests.get(data["image_url"], timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))

        vector = generate_embedding(np.array(img))

        db.collection("Shops") \
            .document(data["shop_id"]) \
            .collection("categories") \
            .document(data["category_id"]) \
            .collection("items") \
            .document(data["item_id"]) \
            .collection("embeddings") \
            .document(str(data["image_index"])) \
            .set({
                "vector": vector.tolist(),
                "model": "mobilenet_v2_100_224",
                "updatedAt": firestore.SERVER_TIMESTAMP,
            })

        return jsonify({
            "status": "success",
            "embedding_length": len(vector),
        })

    except Exception as e:
        print("🔥 /vectorize-item error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================================
# ======================================================
# BATCH-AWARE SALES SEARCH ROUTE WITH FIXED CONVERSION LOGIC
@app.route("/sales", methods=["POST"])
def sales():
    """
    SUPER SMART BATCH-AWARE SALES SEARCH WITH AUTO-SWITCHING
    WITH FIXED CONVERSION LOGIC FOR SELLING UNITS
    """
    try:
        start_time = time.time()
        data = request.get_json() or {}
        
        # Log the incoming request
        print(f"\n{'='*80}")
        print(f"🔍 SEARCH REQUEST RECEIVED at {time.strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"📋 Request Data: {json.dumps(data, indent=2)}")

        # Get query and shop_id
        query = (data.get("query") or "").lower().strip()
        shop_id = data.get("shop_id")
        customer_cart_id = data.get("cart_id")
        
        print(f"🎯 Search Query: '{query}' (original: '{data.get('query', '')}')")
        print(f"🏪 Shop ID: {shop_id}")
        print(f"🛒 Cart ID: {customer_cart_id}")

        if not query or not shop_id:
            error_msg = f"Missing {'query' if not query else ''}{' and ' if not query and not shop_id else ''}{'shop_id' if not shop_id else ''}"
            print(f"❌ {error_msg}")
            return jsonify({
                "items": [],
                "meta": {
                    "error": error_msg,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }), 400

        # Find shop in cache
        print(f"\n📦 LOOKING FOR SHOP {shop_id} IN CACHE...")
        shop = next((s for s in embedding_cache_full["shops"] if s["shop_id"] == shop_id), None)
        if not shop:
            print(f"❌ Shop {shop_id} NOT FOUND in cache")
            return jsonify({
                "items": [],
                "meta": {
                    "error": f"Shop {shop_id} not found",
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }), 404

        shop_name = shop.get("shop_name", "Unnamed")
        print(f"✅ Found shop: {shop_name}")
        print(f"📊 Shop has {len(shop.get('categories', []))} categories")
        
        results = []
        search_debug_info = []

        # --------------------------------------------------
        # HELPER FUNCTIONS FOR SMART BATCH LOGIC - FIXED!
        # --------------------------------------------------
        
        def get_cart_reservations(item_id, batch_id=None):
            """Get reserved quantities from active carts (simulated)"""
            return 0
        
        def calculate_real_availability(batch, unit_type="base", conversion_factor=1):
            """Calculate REAL available quantity considering cart reservations - FIXED CONVERSION!"""
            batch_qty = float(batch.get("quantity", 0))
            batch_id = batch.get("batch_id")
            item_id = batch.get("item_id", "")
            
            reserved = get_cart_reservations(item_id, batch_id)
            real_available = max(0, batch_qty - reserved)
            
            if unit_type == "selling_unit" and conversion_factor > 0:
                # FIXED: MULTIPLY by conversion_factor, not divide!
                # Example: 1 carton × 10 = 10 Ram sticks available
                available_selling_units = real_available * conversion_factor
                
                # Selling units can be sold as long as there's ANY stock
                can_fulfill_selling_unit = available_selling_units >= 0.000001
                
                return {
                    "real_quantity": real_available,  # In parent units (e.g., cartons)
                    "available_selling_units": available_selling_units,  # In selling units (e.g., Ram sticks)
                    "can_fulfill_base": real_available >= 1,
                    "can_fulfill_selling_unit": can_fulfill_selling_unit,
                    "is_partial": available_selling_units < 1  # For UI display
                }
            else:
                # Base units logic
                return {
                    "real_quantity": real_available,
                    "available_selling_units": 0,
                    "can_fulfill_base": real_available >= 1,
                    "can_fulfill_selling_unit": False,
                    "is_partial": False
                }
        
        def find_best_batch_for_unit(batches, unit_type, conversion_factor=1, current_batch_id=None):
            """Find the best batch for a specific unit type"""
            if not batches:
                return None, []
            
            sorted_batches = sorted(batches, key=lambda b: b.get("timestamp", 0))
            best_batch = None
            alternative_batches = []
            
            for batch in sorted_batches:
                availability = calculate_real_availability(batch, unit_type, conversion_factor)
                
                is_current_batch = (current_batch_id == batch.get("batch_id"))
                
                if unit_type == "base":
                    can_fulfill = availability["can_fulfill_base"]
                else:
                    can_fulfill = availability["can_fulfill_selling_unit"]
                
                batch_info = {
                    "batch": batch,
                    "availability": availability,
                    "can_fulfill": can_fulfill,
                    "is_current": is_current_batch,
                    "available_selling_units": availability.get("available_selling_units", 0)
                }
                
                if is_current_batch and can_fulfill:
                    return batch_info, alternative_batches
                
                if can_fulfill and not best_batch:
                    best_batch = batch_info
                else:
                    alternative_batches.append(batch_info)
            
            # If no batch can fulfill, return the first batch anyway
            if not best_batch and sorted_batches:
                first_batch = sorted_batches[0]
                availability = calculate_real_availability(first_batch, unit_type, conversion_factor)
                best_batch = {
                    "batch": first_batch,
                    "availability": availability,
                    "can_fulfill": availability.get("can_fulfill_selling_unit", False),
                    "is_current": False,
                    "is_fallback": True,
                    "available_selling_units": availability.get("available_selling_units", 0)
                }
                print(f"        🔄 Using fallback batch")
            
            return best_batch, alternative_batches
        
        def generate_notifications(batch_info, unit_type, conversion_factor=1):
            """Generate smart notifications for batch"""
            notifications = []
            batch = batch_info.get("batch", {})
            availability = batch_info.get("availability", {})
            
            if not batch:
                return notifications
            
            if unit_type == "base":
                # Base unit notifications
                real_qty = availability.get("real_quantity", 0)
                if 0 < real_qty < 5:
                    notifications.append({
                        "type": "low_stock_warning",
                        "message": f"Only {real_qty:.1f} base units left in '{batch.get('batch_name', 'current')}' batch",
                        "severity": "warning"
                    })
            else:
                # Selling unit notifications
                available_units = availability.get("available_selling_units", 0)
                if available_units > 0:
                    if available_units < 3:
                        notifications.append({
                            "type": "low_stock_warning",
                            "message": f"Only {available_units:.1f} selling units left in '{batch.get('batch_name', 'current')}' batch",
                            "severity": "warning"
                        })
                    
                    # Check if it's a partial unit (less than 1)
                    if 0 < available_units < 1:
                        notifications.append({
                            "type": "partial_stock",
                            "message": f"Partial stock available ({available_units:.2f} units)",
                            "severity": "info"
                        })
            
            # Insufficient quantity warning
            if not batch_info.get("can_fulfill", False):
                if unit_type == "base":
                    notifications.append({
                        "type": "insufficient_for_base",
                        "message": "Not enough for base units (needs ≥1)",
                        "severity": "error",
                        "suggestion": "Try selling units instead"
                    })
                else:
                    # For selling units, check if there's ANY stock
                    if availability.get("available_selling_units", 0) <= 0:
                        notifications.append({
                            "type": "out_of_stock",
                            "message": "Out of stock for selling units",
                            "severity": "error"
                        })
                    else:
                        # There's some stock but maybe not enough
                        notifications.append({
                            "type": "limited_stock",
                            "message": "Limited stock available",
                            "severity": "warning"
                        })
            
            return notifications
        
        # --------------------------------------------------
        # ENHANCED SEARCH SCORING FUNCTION
        # --------------------------------------------------
        
        def calculate_search_score(text, search_query, debug_name=""):
            """Calculate search relevance score (0-100) with detailed debugging"""
            if not text or not search_query:
                if debug_name:
                    print(f"    {debug_name}: No text or query (score: 0)")
                return 0, []
            
            text_lower = text.lower()
            query_lower = search_query.lower()
            
            debug_steps = []
            
            if text_lower == query_lower:
                debug_steps.append(f"Exact match: '{text}' == '{search_query}'")
                if debug_name:
                    print(f"    {debug_name}: ✅ EXACT MATCH (score: 100)")
                return 100, debug_steps
            
            if text_lower.startswith(query_lower):
                debug_steps.append(f"Starts with query: '{text}' starts with '{search_query}'")
                if debug_name:
                    print(f"    {debug_name}: ✅ STARTS WITH (score: 90)")
                return 90, debug_steps
            
            words = text_lower.split()
            for word in words:
                if word.startswith(query_lower):
                    debug_steps.append(f"Word starts with: word '{word}' in '{text}' starts with '{search_query}'")
                    if debug_name:
                        print(f"    {debug_name}: ✅ WORD STARTS WITH (score: 85)")
                    return 85, debug_steps
            
            padded_text = f" {text_lower} "
            padded_query = f" {query_lower} "
            if padded_query in padded_text:
                debug_steps.append(f"Whole word match: '{search_query}' found as whole word in '{text}'")
                if debug_name:
                    print(f"    {debug_name}: ✅ WHOLE WORD MATCH (score: 80)")
                return 80, debug_steps
            
            if query_lower in text_lower:
                position = text_lower.find(query_lower)
                position_penalty = min(position * 0.5, 10)
                score = max(70, 79 - position_penalty)
                debug_steps.append(f"Partial match at position {position}: '{search_query}' found in '{text}' (penalty: {position_penalty:.1f})")
                if debug_name:
                    print(f"    {debug_name}: ✅ PARTIAL MATCH at position {position} (score: {score:.1f})")
                return score, debug_steps
            
            debug_steps.append(f"No match: '{search_query}' not found in '{text}'")
            if debug_name:
                print(f"    {debug_name}: ❌ NO MATCH (score: 0)")
            return 0, debug_steps

        # --------------------------------------------------
        # IMPROVED SEARCH LOGIC
        # --------------------------------------------------
        
        print(f"\n🔍 SEARCHING ACROSS SHOP '{shop_name}'...")
        total_items_scanned = 0
        total_selling_units_scanned = 0
        
        for category_idx, category in enumerate(shop.get("categories", [])):
            category_id = category.get("category_id")
            category_name = category.get("category_name")
            
            print(f"\n  📂 Category {category_idx+1}: {category_name} (ID: {category_id})")
            print(f"  {'─'*60}")

            for item_idx, item in enumerate(category.get("items", [])):
                item_name = item.get("name", "")
                item_name_lower = item_name.lower()
                item_id = item.get("item_id")
                batches = item.get("batches", [])
                
                if not batches:
                    continue
                
                total_items_scanned += 1
                
                print(f"\n    📍 Item {item_idx+1}: '{item_name}' (ID: {item_id})")
                print(f"      Has {len(batches)} batch(es), {len(item.get('selling_units', []))} selling unit(s)")
                
                batches = item.get("batches", [])
                if not batches:
                    print(f"      ⚠️  Skipping - no batches")
                    continue
                
                current_batch_id = None
                
                # --------------------------------------------------
                # PROCESS MAIN ITEM (BASE UNITS)
                # --------------------------------------------------
                print(f"      🔍 Checking main item match...")
                main_item_score, main_item_debug = calculate_search_score(
                    item_name, query, f"Main Item '{item_name}'"
                )
                main_item_matches = main_item_score > 0
                
                if main_item_matches:
                    print(f"      ✅ MAIN ITEM MATCHED with score {main_item_score}")
                    
                    best_batch_info, alternative_batches = find_best_batch_for_unit(
                        batches, "base", current_batch_id=current_batch_id
                    )
                    
                    if best_batch_info:
                        batch = best_batch_info["batch"]
                        availability = best_batch_info["availability"]
                        notifications = generate_notifications(best_batch_info, "base")
                        
                        real_qty = availability["real_quantity"]
                        if real_qty >= 1:
                            batch_status = "active_healthy" if real_qty > 3 else "active_low_stock"
                        elif real_qty > 0:
                            batch_status = "insufficient_for_base"
                        else:
                            batch_status = "exhausted"
                        
                        next_available_batch = None
                        for alt in alternative_batches:
                            if alt.get("can_fulfill", False):
                                next_available_batch = alt["batch"]
                                break
                        
                        main_item_response = {
                            "type": "main_item",
                            "item_id": item_id,
                            "main_item_id": item_id,
                            "category_id": item.get("category_id") or category_id,
                            "category_name": item.get("category_name") or category_name,
                            "name": item_name,
                            "display_name": item_name,
                            "thumbnail": item.get("thumbnail"),
                            "batch_status": batch_status,
                            "batch_id": batch.get("batch_id"),
                            "batch_name": batch.get("batch_name"),
                            "batch_remaining": availability["real_quantity"],
                            "real_available": availability["real_quantity"],
                            "price": round(float(batch.get("sell_price", 0)), 2),
                            "base_unit": batch.get("unit", item.get("base_unit", "unit")),
                            "batch_switch_required": not best_batch_info.get("can_fulfill", False),
                            "can_fulfill": best_batch_info.get("can_fulfill", False),
                            "is_current_batch": best_batch_info.get("is_current", False),
                            "next_batch_available": next_available_batch is not None,
                            "next_batch_id": next_available_batch.get("batch_id") if next_available_batch else None,
                            "next_batch_name": next_available_batch.get("batch_name") if next_available_batch else None,
                            "next_batch_price": round(float(next_available_batch.get("sell_price", 0)), 2) if next_available_batch else None,
                            "notifications": notifications,
                            "unit_type": "base",
                            "search_score": main_item_score,
                            "parent_item_name": item_name,
                            "debug": {
                                "match_type": "main_item_direct",
                                "matched_text": item_name,
                                "score_calculation": main_item_debug,
                                "query_used": query,
                                "batch_availability": real_qty
                            }
                        }
                        results.append(main_item_response)
                        
                        search_debug_info.append({
                            "item_name": item_name,
                            "type": "main_item",
                            "score": main_item_score,
                            "batch_status": batch_status,
                            "can_fulfill": best_batch_info.get("can_fulfill", False)
                        })
                        
                        print(f"      📝 Added to results (score: {main_item_score}, batch: {batch_status})")
                    else:
                        print(f"      ⚠️  No suitable batch found")
                else:
                    print(f"      ❌ No match for main item")

                # --------------------------------------------------
                # PROCESS SELLING UNITS WITH CORRECTED CONVERSION
                # --------------------------------------------------
                selling_units = item.get("selling_units", [])
                total_selling_units_scanned += len(selling_units)
                
                if selling_units:
                    print(f"      🔍 Checking {len(selling_units)} selling unit(s)...")
                
                for su_idx, su in enumerate(selling_units):
                    su_name = su.get("name", "")
                    su_display_name = su.get("display_name", su_name)
                    
                    su_scores = []
                    su_debug_info = []
                    
                    su_name_score, su_name_debug = calculate_search_score(
                        su_name, query, f"SU Name '{su_name}'"
                    )
                    if su_name_score > 0:
                        su_scores.append(("su_name", su_name_score))
                        su_debug_info.extend([f"SU Name: {d}" for d in su_name_debug])
                    
                    su_display_score, su_display_debug = calculate_search_score(
                        su_display_name, query, f"SU Display '{su_display_name}'"
                    )
                    if su_display_score > 0:
                        su_scores.append(("su_display", su_display_score))
                        su_debug_info.extend([f"SU Display: {d}" for d in su_display_debug])
                    
                    parent_item_score, parent_debug = calculate_search_score(item_name, query, f"Parent '{item_name}'")
                    if parent_item_score > 50:
                        inherited_score = parent_item_score * 0.7
                        su_scores.append(("parent_inherited", inherited_score))
                        su_debug_info.extend([f"Parent Inheritance: {d} (inherited: {inherited_score:.1f})" for d in parent_debug])
                    
                    if su_scores:
                        best_score_type, max_score = max(su_scores, key=lambda x: x[1])
                        
                        if max_score > 30:
                            print(f"      ✅ Selling Unit {su_idx+1}: '{su_display_name}' matched via {best_score_type} (score: {max_score:.1f})")
                            
                            conversion = float(su.get("conversion_factor", 1))
                            if conversion <= 0:
                                print(f"      ⚠️  Skipping - invalid conversion factor: {conversion}")
                                continue
                            
                            # Find the best batch for this selling unit
                            best_batch_info, alternative_batches = find_best_batch_for_unit(
                                batches, "selling_unit", conversion, current_batch_id
                            )
                            
                            batch = None
                            availability = None
                            can_fulfill = False
                            batch_status = "no_suitable_batch"
                            notifications = []
                            unit_price = 0
                            available_selling_units = 0
                            
                            if best_batch_info:
                                batch = best_batch_info["batch"]
                                availability = best_batch_info["availability"]
                                notifications = generate_notifications(best_batch_info, "selling_unit", conversion)
                                can_fulfill = best_batch_info.get("can_fulfill", False)
                                available_selling_units = availability.get("available_selling_units", 0)
                                
                                # Determine batch status
                                if available_selling_units >= 1:
                                    batch_status = "active_healthy" if available_selling_units > 10 else "active_low_stock"
                                elif available_selling_units > 0:
                                    batch_status = "partial_stock"
                                else:
                                    batch_status = "out_of_stock"
                                
                                # Calculate price per selling unit
                                if batch and conversion > 0:
                                    unit_price = float(batch.get("sell_price", 0)) / conversion
                                
                                print(f"        ✅ Found batch: {batch.get('batch_name', 'unnamed')}")
                                print(f"        📊 Available selling units: {available_selling_units} (conversion: {conversion})")
                            else:
                                print(f"        ⚠️  No suitable batch found, showing anyway")
                                
                                if batches:
                                    # Use first batch for display purposes
                                    first_batch = sorted(batches, key=lambda b: b.get("timestamp", 0))[0]
                                    batch = first_batch
                                    availability = calculate_real_availability(first_batch, "selling_unit", conversion)
                                    available_selling_units = availability.get("available_selling_units", 0)
                                    
                                    if conversion > 0 and batch.get("sell_price"):
                                        unit_price = float(batch.get("sell_price", 0)) / conversion
                                    
                                    notifications = [{
                                        "type": "no_batch_link",
                                        "message": "No batch link configured",
                                        "severity": "warning"
                                    }]
                                    batch_status = "no_batch_link"
                                else:
                                    notifications = [{
                                        "type": "no_batches",
                                        "message": "No stock batches available",
                                        "severity": "error"
                                    }]
                                    batch_status = "no_batches"
                            
                            # Find next available batch
                            next_available_batch = None
                            if alternative_batches:
                                for alt in alternative_batches:
                                    if alt.get("can_fulfill", False):
                                        next_available_batch = alt["batch"]
                                        break
                            
                            next_unit_price = None
                            if next_available_batch and conversion > 0:
                                next_unit_price = float(next_available_batch.get("sell_price", 0)) / conversion
                            
                            # Create selling unit response
                            selling_unit_response = {
                                "type": "selling_unit",
                                "item_id": item_id,
                                "main_item_id": item_id,
                                "sell_unit_id": su.get("sell_unit_id"),
                                "category_id": item.get("category_id") or category_id,
                                "category_name": item.get("category_name") or category_name,
                                "name": f"{su_name}",
                                "display_name": su_display_name,
                                "parent_item_name": item_name,
                                "thumbnail": su.get("thumbnail") or item.get("thumbnail"),
                                "batch_status": batch_status,
                                "batch_id": batch.get("batch_id") if batch else None,
                                "batch_name": batch.get("batch_name") if batch else None,
                                "batch_remaining": availability["real_quantity"] if availability else 0,
                                "real_available_units": available_selling_units,  # This is now CORRECT!
                                "real_available_fraction": 0,  # Not used with new logic
                                "price": round(unit_price, 4),
                                "available_stock": round(float(batch.get("quantity", 0)) if batch else 0, 2),
                                "conversion_factor": conversion,
                                "base_unit": batch.get("unit", item.get("base_unit", "unit")) if batch else item.get("base_unit", "unit"),
                                "batch_switch_required": not can_fulfill and available_selling_units <= 0,
                                "can_fulfill": can_fulfill,
                                "is_current_batch": best_batch_info.get("is_current", False) if best_batch_info else False,
                                "next_batch_available": next_available_batch is not None,
                                "next_batch_id": next_available_batch.get("batch_id") if next_available_batch else None,
                                "next_batch_name": next_available_batch.get("batch_name") if next_available_batch else None,
                                "next_batch_price": round(next_unit_price, 4) if next_unit_price else None,
                                "has_batch_links": len(su.get("batch_links", [])) > 0,
                                "batch_links": su.get("batch_links", []),
                                "notifications": notifications,
                                "unit_type": "selling_unit",
                                "search_score": max_score,
                                "matched_by": best_score_type,
                                "debug": {
                                    "match_type": best_score_type,
                                    "matched_text": su_display_name if best_score_type == "su_display" else su_name,
                                    "score_calculation": su_debug_info,
                                    "parent_item": item_name,
                                    "parent_score": parent_item_score,
                                    "query_used": query,
                                    "batch_available_units": available_selling_units,
                                    "conversion_applied": conversion,
                                    "parent_batch_qty": batch.get("quantity", 0) if batch else 0
                                }
                            }
                            results.append(selling_unit_response)
                            
                            search_debug_info.append({
                                "item_name": f"{item_name} → {su_display_name}",
                                "type": "selling_unit",
                                "score": max_score,
                                "match_type": best_score_type,
                                "batch_status": batch_status,
                                "can_fulfill": can_fulfill,
                                "available_units": available_selling_units,
                                "conversion": conversion
                            })
                            
                            print(f"      📝 Added selling unit (score: {max_score:.1f}, status: {batch_status}, units: {available_selling_units})")
                        else:
                            print(f"      ❌ Selling unit score too low: {max_score:.1f} (threshold: 30)")
                    else:
                        if len(selling_units) <= 3:
                            print(f"      ❌ Selling Unit {su_idx+1}: '{su_display_name}' - no match")

        # --------------------------------------------------
        # ENHANCED SORTING WITH SEARCH SCORING
        # --------------------------------------------------
        
        print(f"\n📊 SORTING {len(results)} RESULTS...")
        
        # Sort with priority:
        # 1. Can fulfill (available for sale)
        # 2. Higher search score
        # 3. More available units
        # 4. Main items before selling units
        # 5. Alphabetical
        results.sort(key=lambda x: (
            not x.get("can_fulfill", False),
            -x.get("search_score", 0),
            -x.get("real_available_units", 0),
            x.get("type") == "selling_unit",
            x.get("name", "").lower()
        ))
        
        print(f"\n🏆 FINAL RESULTS ORDER:")
        for i, result in enumerate(results[:10]):
            print(f"  {i+1}. {result.get('type')}: '{result.get('name')}'")
            print(f"     Score: {result.get('search_score', 0):.1f}, Can fulfill: {result.get('can_fulfill')}")
            print(f"     Available units: {result.get('real_available_units', 0)}")
            print(f"     Batch: {result.get('batch_status')}")

        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Calculate statistics
        scored_results = len([r for r in results if r.get("search_score", 0) > 0])
        high_score_results = len([r for r in results if r.get("search_score", 0) >= 80])
        main_items_count = len([r for r in results if r.get("type") == "main_item"])
        selling_units_count = len([r for r in results if r.get("type") == "selling_unit"])
        
        can_fulfill_count = sum(1 for r in results if r.get("can_fulfill", False))
        needs_switch_count = sum(1 for r in results if r.get("batch_switch_required", False))

        print(f"\n{'='*80}")
        print(f"📈 SEARCH COMPLETE WITH FIXED CONVERSION LOGIC")
        print(f"{'='*80}")
        print(f"Total items scanned: {total_items_scanned}")
        print(f"Total selling units scanned: {total_selling_units_scanned}")
        print(f"Total results found: {len(results)}")
        print(f"  - Main items: {main_items_count}")
        print(f"  - Selling units: {selling_units_count}")
        print(f"  - Can fulfill orders: {can_fulfill_count}")
        print(f"  - Need batch switch: {needs_switch_count}")
        print(f"  - High score matches (≥80): {high_score_results}")
        print(f"Processing time: {processing_time}ms")
        print(f"{'='*80}\n")

        return jsonify({
            "items": results,
            "meta": {
                "shop_id": shop_id,
                "shop_name": shop_name,
                "query": query,
                "cart_id": customer_cart_id,
                "results": len(results),
                "scored_results": scored_results,
                "high_score_results": high_score_results,
                "main_items_count": main_items_count,
                "selling_units_count": selling_units_count,
                "can_fulfill_count": can_fulfill_count,
                "needs_switch_count": needs_switch_count,
                "items_scanned": total_items_scanned,
                "selling_units_scanned": total_selling_units_scanned,
                "processing_time_ms": processing_time,
                "cache_last_updated": embedding_cache_full.get("last_updated"),
                "note": "Enhanced search with FIXED conversion logic (multiply, not divide!)"
            },
            "debug": {
                "search_debug_info": search_debug_info,
                "sorting_priority": [
                    "1. Items that can fulfill orders",
                    "2. Higher search score",
                    "3. More available units",
                    "4. Main items before selling units",
                    "5. Alphabetical order"
                ],
                "conversion_logic": [
                    "Selling units: available = parent_quantity × conversion_factor",
                    "Price per unit: price = batch_price ÷ conversion_factor",
                    "Can fulfill if: available_selling_units > 0"
                ]
            }
        }), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ UNEXPECTED ERROR:")
        print(f"{'='*80}")
        print(error_trace)
        print(f"{'='*80}")
        
        return jsonify({
            "items": [],
            "meta": {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "note": "Check server logs for detailed error trace"
            }
        }), 500
# ======================================================
@app.route("/complete-sale", methods=["POST"])
def complete_sale():
    """
    COMPLETE SALE (FIXED CONVERSION LOGIC)
    """
    try:
        data = request.get_json(force=True)
        shop_id = data.get("shop_id")
        seller = data.get("seller")
        items = data.get("items", [])

        if not shop_id or not items:
            return jsonify({"success": False, "error": "Missing shop_id or items"}), 400

        updated_items = []

        print("\n🔥 COMPLETE SALE REQUEST")
        print(f"Shop ID: {shop_id} | Items: {len(items)}")

        for idx, cart_item in enumerate(items):
            print(f"\n📦 Processing item {idx + 1}")
            
            item_id = cart_item.get("item_id")
            category_id = cart_item.get("category_id")
            batch_id = cart_item.get("batch_id") or cart_item.get("batchId")
            quantity = float(cart_item.get("quantity", 0))
            unit = cart_item.get("unit", "unit")
            conversion_factor = float(cart_item.get("conversion_factor", 1))
            item_type = cart_item.get("type", "main_item")  # main_item or selling_unit
            
            print(f"   Type: {item_type}")
            print(f"   Quantity entered: {quantity}")
            print(f"   Conversion factor: {conversion_factor}")

            if not item_id or not category_id or not batch_id or quantity <= 0:
                return jsonify({
                    "success": False,
                    "error": "Invalid sale item payload",
                    "item": cart_item
                }), 400

            # Firestore path to item
            item_ref = (
                db.collection("Shops")
                .document(shop_id)
                .collection("categories")
                .document(category_id)
                .collection("items")
                .document(item_id)
            )

            item_doc = item_ref.get()
            if not item_doc.exists:
                return jsonify({
                    "success": False,
                    "error": f"Item {item_id} not found"
                }), 404

            item_data = item_doc.to_dict()
            batches = item_data.get("batches", [])
            total_stock = float(item_data.get("stock", 0))

            # Find the target batch
            batch_index = next((i for i, b in enumerate(batches) if b.get("id") == batch_id), None)
            if batch_index is None:
                return jsonify({
                    "success": False,
                    "error": f"Batch {batch_id} not found for item {item_data.get('name')}"
                }), 404

            batch = batches[batch_index]
            batch_qty = float(batch.get("quantity", 0))

            # ✅ CRITICAL FIX: CONVERSION LOGIC
            if item_type == "selling_unit":
                # Selling units: quantity ÷ conversion_factor
                # Example: 2 single units ÷ 20 = 0.1 cartons
                base_qty = quantity / conversion_factor
                print(f"   Selling unit: {quantity} units ÷ {conversion_factor} = {base_qty} base units")
            else:
                # Main item: no conversion needed
                base_qty = quantity
                print(f"   Main item: {quantity} base units")

            print(f"   Batch available: {batch_qty} base units")
            print(f"   Required to deduct: {base_qty} base units")

            if batch_qty < base_qty:
                return jsonify({
                    "success": False,
                    "error": f"Insufficient stock in batch {batch_id}. Available: {batch_qty} base units, requested: {base_qty} base units",
                    "details": {
                        "item_type": item_type,
                        "quantity_requested": quantity,
                        "conversion_factor": conversion_factor,
                        "base_units_needed": base_qty,
                        "base_units_available": batch_qty
                    }
                }), 400

            # Deduct stock
            batches[batch_index]["quantity"] = batch_qty - base_qty
            new_total_stock = total_stock - base_qty

            # Calculate price
            sell_price = float(batch.get("sellPrice", 0))
            if item_type == "selling_unit":
                # Price per selling unit = batch sell price ÷ conversion_factor
                unit_price = sell_price / conversion_factor
                total_price = unit_price * quantity
            else:
                total_price = sell_price * base_qty

            # Create stock transaction
            stock_txn = {
                "id": f"sale_{int(time.time() * 1000)}",
                "type": "sale",
                "item_type": item_type,
                "batchId": batch_id,
                "quantity": base_qty,  # In base units
                "selling_units_quantity": quantity if item_type == "selling_unit" else None,
                "unit": unit,
                "sellPrice": sell_price,
                "unitPrice": unit_price if item_type == "selling_unit" else sell_price,
                "totalPrice": total_price,
                "timestamp": int(datetime.now().timestamp()),
                "performedBy": seller,
                "conversion_factor": conversion_factor if item_type == "selling_unit" else None
            }

            stock_transactions = item_data.get("stockTransactions", [])
            stock_transactions.append(stock_txn)

            # Update Firestore
            item_ref.update({
                "batches": batches,
                "stock": new_total_stock,
                "stockTransactions": stock_transactions,
                "lastStockUpdate": firestore.SERVER_TIMESTAMP,
                "lastTransactionId": stock_txn["id"]
            })

            exhausted = batches[batch_index]["quantity"] == 0

            updated_items.append({
                "item_id": item_id,
                "item_type": item_type,
                "batch_id": batch_id,
                "quantity_sold": quantity,
                "base_units_deducted": base_qty,
                "remaining_batch_quantity": batches[batch_index]["quantity"],
                "remaining_total_stock": new_total_stock,
                "batch_exhausted": exhausted,
                "total_price": total_price
            })

            print(f"   ✅ Deducted: {base_qty} base units from batch")
            print(f"   ✅ Remaining in batch: {batches[batch_index]['quantity']}")
            print(f"   ✅ Total price: ${total_price}")

        return jsonify({
            "success": True,
            "updated_items": updated_items,
            "message": "Sale completed successfully"
        }), 200

    except Exception as e:
        print("🔥 COMPLETE SALE ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ======================================================
# ITEM OPTIMIZATION (UPDATED WITH BATCH INFO)
# ======================================================
@app.route("/item-optimization", methods=["GET"])
def item_optimization():
    # Calculate batch statistics
    total_batches = 0
    items_with_batches = 0
    items_without_batches = 0
    
    for shop in embedding_cache_full["shops"]:
        for category in shop["categories"]:
            for item in category["items"]:
                if item.get("has_batches"):
                    items_with_batches += 1
                    total_batches += len(item.get("batches", []))
                else:
                    items_without_batches += 1
    
    return jsonify({
        "status": "success",
        "shops": embedding_cache_full["shops"],
        "total_shops": embedding_cache_full["total_shops"],
        "last_updated": embedding_cache_full["last_updated"],
        "batch_stats": {
            "total_batches": total_batches,
            "items_with_batches": items_with_batches,
            "items_without_batches": items_without_batches,
            "percentage_with_batches": round(items_with_batches / (items_with_batches + items_without_batches) * 100, 1) if (items_with_batches + items_without_batches) > 0 else 0
        }
    })


# ======================================================
# DEBUG ENDPOINT (UPDATED WITH BATCH INFO)
# ======================================================
@app.route("/debug-cache", methods=["GET"])
def debug_cache():
    """Debug endpoint to check cache contents (updated with batch tracking)"""
    if not embedding_cache_full["shops"]:
        return jsonify({"error": "Cache empty"}), 404
    
    try:
        first_shop = embedding_cache_full["shops"][0]
        first_category = first_shop["categories"][0]
        first_item = first_category["items"][0]
        
        # Count statistics
        total_selling_units = 0
        total_batches = 0
        items_with_batches = 0
        
        for shop in embedding_cache_full["shops"]:
            for category in shop["categories"]:
                for item in category["items"]:
                    total_selling_units += len(item.get("selling_units", []))
                    total_batches += len(item.get("batches", []))
                    if item.get("has_batches"):
                        items_with_batches += 1
        
        return jsonify({
            "first_item": {
                "name": first_item["name"],
                "has_sell_price": "sell_price" in first_item or "sellPrice" in first_item,
                "sell_price_value": first_item.get("sell_price") or first_item.get("sellPrice"),
                "has_batches": first_item.get("has_batches", False),
                "batch_count": len(first_item.get("batches", [])),
                "has_selling_units": len(first_item.get("selling_units", [])) > 0,
                "selling_units_count": len(first_item.get("selling_units", []))
            },
            "cache_details": {
                "total_shops": len(embedding_cache_full["shops"]),
                "total_categories": sum(len(shop["categories"]) for shop in embedding_cache_full["shops"]),
                "total_items": sum(len(category["items"]) for shop in embedding_cache_full["shops"] for category in shop["categories"]),
                "total_selling_units": total_selling_units,
                "total_batches": total_batches,
                "items_with_batches": items_with_batches,
                "last_updated": embedding_cache_full["last_updated"]
            }
        })
    except (IndexError, KeyError) as e:
        return jsonify({"error": f"Cache structure issue: {str(e)}"}), 500





# ======================================================
# PLAN INITIALIZATION ROUTES
# ======================================================
@app.route("/ensure-plan", methods=["POST"])
def ensure_plan():
    """
    Ensure a default plan exists for a given shop.
    Creates a 'Solo' plan only if none exists.
    """
    try:
        data = request.get_json(silent=True) or {}
        shop_id = data.get("shop_id")

        if not shop_id:
            return jsonify({
                "success": False,
                "error": "shop_id is required"
            }), 400

        plan_ref = (
            db.collection("Shops")
              .document(shop_id)
              .collection("plan")
              .document("default")
        )

        if plan_ref.get().exists:
            return jsonify({
                "success": True,
                "message": "Plan already exists for this shop."
            })

        default_plan = {
            "name": "Solo",
            "staffLimit": 0,
            "features": {
                "sell": True,
                "manageStock": True,
                "businessIntelligence": False,
                "settings": True
            },
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP
        }

        plan_ref.set(default_plan)

        print(f"✅ Default plan initialized for shop: {shop_id}")

        return jsonify({
            "success": True,
            "message": "Default plan initialized successfully."
        })

    except Exception as e:
        print(f"🔥 ensure-plan error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


# ======================================================
# ADMIN DASHBOARD
# ======================================================
@app.route("/admin")
def admin():
    return render_template("admindashboard.html")


# ======================================================
# TEST SELLING UNITS ENDPOINT
# ======================================================
@app.route("/test-selling-units", methods=["GET"])
def test_selling_units():
    """Test endpoint to check selling units directly in Firestore"""
    try:
        shop_id = request.args.get("shop_id")
        item_id = request.args.get("item_id")
        
        if not shop_id or not item_id:
            return jsonify({"error": "shop_id and item_id required"}), 400
        
        # Find the item in Firestore
        items_ref = db.collection("Shops").document(shop_id).collection("items").document(item_id)
        item_doc = items_ref.get()
        
        if not item_doc.exists:
            return jsonify({"error": "Item not found"}), 404
        
        item_data = item_doc.to_dict()
        
        # Try to get selling units
        sell_units_ref = items_ref.collection("sellUnits")
        sell_units_docs = list(sell_units_ref.stream())
        
        result = {
            "item_name": item_data.get("name"),
            "item_id": item_id,
            "sellUnits_collection_exists": True,
            "sellUnits_count": len(sell_units_docs),
            "sellUnits_details": []
        }
        
        for doc in sell_units_docs:
            data = doc.to_dict()
            result["sellUnits_details"].append({
                "id": doc.id,
                "name": data.get("name"),
                "conversionFactor": data.get("conversionFactor"),
                "sellPrice": data.get("sellPrice"),
                "has_batchLinks": "batchLinks" in data,
                "batchLinks_count": len(data.get("batchLinks", []))
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# STARTUP INITIALIZATION (SAFE) AND RUNNING THE SERVER
# ======================================================
has_initialized = False

def startup_init():
    global has_initialized
    if has_initialized:
        return

    print("[INIT] Preloading FULL cache (with batch tracking)...")
    print("[NOTE] Embedding/vectorization features are disabled")
    refresh_full_item_cache()

    print("[INIT] Setting up Firestore listeners...")
    db.collection_group("items").on_snapshot(on_full_item_snapshot)
    db.collection_group("sellUnits").on_snapshot(on_selling_units_snapshot)

    print("[READY] Listeners active for items and selling units")
    print("[READY] App running without embedding/ML dependencies")

    has_initialized = True
# Render / Gunicorn
if os.environ.get("RENDER") == "true":
    startup_init()

# Local development
if __name__ == "__main__":
    startup_init()
    app.run(debug=True)




