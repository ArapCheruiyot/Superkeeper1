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
cred = credentials.Certificate("FIREBASE_KEY")
firebase_admin.initialize_app(cred)
db = firestore.client()


# ======================================================
# TODO: LOAD MODEL - Currently disabled for safe deployment
# ======================================================
# print("[INIT] Loading TensorFlow Hub model...")
# model = hub.load(
#     "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
# )
# print("[READY] Model loaded successfully.")
model = None  # Placeholder


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

                # TODO: Re-enable embeddings when needed
                # Get embeddings (if any)
                # embeddings = []
                # for emb_doc in item_doc.reference.collection("embeddings").stream():
                #     vector = emb_doc.to_dict().get("vector")
                #     if vector:
                #         embeddings.append(np.array(vector))
                embeddings = []  # Placeholder - no embeddings for now

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
                    "has_embeddings": False,  # TODO: Set to len(embeddings) > 0 when re-enabling embeddings
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

# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ======================================================
# TODO: VECTORIZE ITEM (STOCK IMAGE → EMBEDDING) - Currently disabled
# ======================================================
@app.route("/vectorize-item", methods=["POST"])
def vectorize_item():
    """
    TODO: Re-enable image vectorization when needed
    Currently returns placeholder response
    """
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

        print(f"📥 /vectorize-item → {data['item_id']} image {data['image_index']} (DISABLED)")
        
        # TODO: Re-enable image processing and embedding generation
        # response = requests.get(data["image_url"], timeout=10)
        # img = Image.open(BytesIO(response.content)).convert("RGB")
        # img = img.resize((224, 224))
        # vector = generate_embedding(np.array(img))
        
        # Placeholder: Return success without actual processing
        return jsonify({
            "status": "success",
            "embedding_length": 0,
            "note": "Vectorization disabled - placeholder response"
        })

    except Exception as e:
        print("🔥 /vectorize-item error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================================
# BATCH-AWARE SALES SEARCH ROUTE (READ-ONLY)
@app.route("/sales", methods=["POST"])
def sales():
    """
    ENHANCED BATCH-AWARE SALES SEARCH - FIXED FOR SELLING UNITS
    """
    try:
        start_time = time.time()
        data = request.get_json() or {}

        # Simplified logging for speed
        query = (data.get("query") or "").lower().strip()
        shop_id = data.get("shop_id")

        if not query or not shop_id:
            return jsonify({
                "items": [],
                "meta": {
                    "error": "Missing query or shop_id",
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }), 400

        # Find shop in cache
        shop = next((s for s in embedding_cache_full["shops"] if s["shop_id"] == shop_id), None)
        if not shop:
            return jsonify({
                "items": [],
                "meta": {
                    "error": f"Shop {shop_id} not found",
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }), 404

        shop_name = shop.get("shop_name", "Unnamed")
        results = []

        # --------------------------------------------------
        # IMPROVED SEARCH LOGIC
        # --------------------------------------------------
        for category in shop.get("categories", []):
            category_id = category.get("category_id")
            category_name = category.get("category_name")

            for item in category.get("items", []):
                item_name = item.get("name", "")
                item_name_lower = item_name.lower()
                item_id = item.get("item_id")
                
                # Check if main item matches query
                main_item_matches = query in item_name_lower
                
                # Process batches for this item
                batches = item.get("batches", [])
                sorted_batches = sorted(batches, key=lambda b: b.get("timestamp", 0))

                active_batch = None
                next_batch = None
                for i, batch in enumerate(sorted_batches):
                    if float(batch.get("quantity", 0)) > 0.001:
                        active_batch = batch
                        next_batch = sorted_batches[i + 1] if i + 1 < len(sorted_batches) else None
                        break

                if active_batch:
                    qty = float(active_batch.get("quantity", 0))
                    if qty > 1:
                        batch_status = "active_healthy"
                    elif qty > 0:
                        batch_status = "active_last_item"
                    else:
                        batch_status = "exhausted"
                elif batches:
                    batch_status = "all_exhausted"
                else:
                    batch_status = "no_stock"

                # --------------------------------------------------
                # ADD MAIN ITEM IF IT MATCHES
                # --------------------------------------------------
                if main_item_matches:
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
                        "batch_id": active_batch.get("batch_id") if active_batch else None,
                        "batch_name": active_batch.get("batch_name") if active_batch else None,
                        "batch_remaining": float(active_batch.get("quantity", 0)) if active_batch else 0,
                        "price": round(float(active_batch.get("sell_price", 0)), 2) if active_batch else 0,
                        "base_unit": active_batch.get("unit", item.get("base_unit", "unit")) if active_batch else item.get("base_unit", "unit"),
                        "batch_switch_required": batch_status in ["exhausted", "all_exhausted"],
                        "next_batch_available": next_batch is not None,
                        "next_batch_id": next_batch.get("batch_id") if next_batch else None,
                        "next_batch_name": next_batch.get("batch_name") if next_batch else None,
                        "next_batch_price": round(float(next_batch.get("sell_price", 0)), 2) if next_batch else None,
                        "next_batch_remaining": float(next_batch.get("quantity", 0)) if next_batch else 0,
                        "unit_type": "base"
                    }
                    results.append(main_item_response)

                # --------------------------------------------------
                # SELLING UNITS - FIXED LOGIC
                # --------------------------------------------------
                for su in item.get("selling_units", []):
                    su_name = su.get("name", "").lower()
                    
                    # FIX: Check if selling unit name matches query, REGARDLESS of main item name
                    if query in su_name:
                        # This selling unit matches the query!
                        conversion = float(su.get("conversion_factor", 1))
                        unit_price = 0
                        available_units = 0

                        if active_batch and conversion > 0:
                            available_units = float(active_batch.get("quantity", 0)) * conversion
                            unit_price = float(active_batch.get("sell_price", 0)) / conversion

                        next_unit_price = None
                        if next_batch and conversion > 0:
                            next_unit_price = float(next_batch.get("sell_price", 0)) / conversion

                        selling_unit_response = {
                            "type": "selling_unit",
                            "item_id": item_id,
                            "main_item_id": item_id,
                            "sell_unit_id": su.get("sell_unit_id"),
                            "category_id": item.get("category_id") or category_id,
                            "category_name": item.get("category_name") or category_name,
                            # Display selling unit prominently
                            "name": f"{su.get('name')}",
                            "display_name": su.get("name"),
                            # Include parent item name in a separate field if needed
                            "parent_item_name": item_name,
                            "thumbnail": su.get("thumbnail") or item.get("thumbnail"),
                            "batch_status": batch_status,
                            "batch_id": active_batch.get("batch_id") if active_batch else None,
                            "batch_remaining": float(active_batch.get("quantity", 0)) if active_batch else 0,
                            "price": round(unit_price, 4),
                            "available_stock": round(available_units, 2),
                            "conversion_factor": conversion,
                            "base_unit": active_batch.get("unit", item.get("base_unit", "unit")) if active_batch else item.get("base_unit", "unit"),
                            "batch_switch_required": batch_status in ["exhausted", "all_exhausted"],
                            "next_batch_available": next_batch is not None,
                            "next_batch_id": next_batch.get("batch_id") if next_batch else None,
                            "next_batch_price": round(next_unit_price, 4) if next_unit_price else None,
                            "next_batch_remaining": float(next_batch.get("quantity", 0)) if next_batch else 0,
                            "has_batch_links": len(su.get("batch_links", [])) > 0,
                            "batch_links": su.get("batch_links", []),
                            "total_units_available": su.get("total_units_available", 0),
                            "unit_type": "selling_unit"
                        }
                        results.append(selling_unit_response)

        # Sort results: selling units that match query exactly first, then main items
        results.sort(key=lambda x: (
            x.get("type") == "selling_unit" and query in x.get("display_name", "").lower(),
            x.get("type") == "selling_unit"
        ), reverse=True)

        processing_time = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "items": results,
            "meta": {
                "shop_id": shop_id,
                "shop_name": shop_name,
                "query": query,
                "results": len(results),
                "processing_time_ms": processing_time,
                "cache_last_updated": embedding_cache_full.get("last_updated"),
                "note": "Embedding-based search disabled - using text search only"
            }
        }), 200

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "items": [],
            "meta": {
                "error": str(e),
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
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
        },
        "note": "Embedding-based optimization disabled"
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
                "selling_units_count": len(first_item.get("selling_units", [])),
                "has_embeddings": first_item.get("has_embeddings", False),  # Will be False since disabled
                "embeddings_count": len(first_item.get("embeddings", []))
            },
            "cache_details": {
                "total_shops": len(embedding_cache_full["shops"]),
                "total_categories": sum(len(shop["categories"]) for shop in embedding_cache_full["shops"]),
                "total_items": sum(len(category["items"]) for shop in embedding_cache_full["shops"] for category in shop["categories"]),
                "total_selling_units": total_selling_units,
                "total_batches": total_batches,
                "items_with_batches": items_with_batches,
                "last_updated": embedding_cache_full["last_updated"],
                "note": "Embedding cache disabled - storing only metadata"
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
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    print("[INIT] Preloading FULL cache (with batch tracking)...")
    print("[NOTE] Embedding/vectorization features are disabled")
    refresh_full_item_cache()
    
    # Set up listeners for both main items AND selling units
    print("[INIT] Setting up Firestore listeners...")
    db.collection_group("items").on_snapshot(on_full_item_snapshot)
    db.collection_group("sellUnits").on_snapshot(on_selling_units_snapshot)
    print("[READY] Listeners active for items and selling units")
    print("[READY] App running without embedding/ML dependencies")
    

    app.run(debug=True)

