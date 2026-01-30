from flask import Flask, render_template, request, jsonify

import requests
import firebase_admin
from firebase_admin import credentials, firestore

import numpy as np
from PIL import Image
from io import BytesIO

import tensorflow_hub as hub
import time
import base64

from embeddings import generate_embedding
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# APP INIT
# ======================================================
app = Flask(__name__)


# ======================================================
# FIREBASE CONFIG
# ======================================================
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# ======================================================
# LOAD MODEL
# ======================================================
print("[INIT] Loading TensorFlow Hub model...")
model = hub.load(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
)
print("[READY] Model loaded successfully.")


# ======================================================
# FULL SHOP CACHE (STRICTLY PER SHOP)
# ======================================================
embedding_cache_full = {
    "shops": [],
    "last_updated": None,
    "total_shops": 0
}


def refresh_full_item_cache():
    start = time.time()
    print("\n[INFO] Refreshing FULL shop cache...")

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

            category_entry = {
                "category_id": cat_doc.id,
                "category_name": cat_data.get("name", ""),
                "items": []
            }

            for item_doc in cat_doc.reference.collection("items").stream():
                item_data = item_doc.to_dict()

                embeddings = []
                for emb_doc in item_doc.reference.collection("embeddings").stream():
                    vector = emb_doc.to_dict().get("vector")
                    if vector:
                        embeddings.append(np.array(vector))

                # ⛔ skip items with NO embeddings
                if not embeddings:
                    continue

                category_entry["items"].append({
                    "item_id": item_doc.id,
                    "name": item_data.get("name", ""),
                    "thumbnail": item_data.get("images", [None])[0],
                    "sellPrice": item_data.get("sellPrice", 0),  # ✅ ADD THIS
                    "buyPrice": item_data.get("buyPrice", 0),   # ✅ Optional
                    "embeddings": embeddings
                })

            # ⛔ skip empty categories
            if category_entry["items"]:
                shop_entry["categories"].append(category_entry)

        # ⛔ skip empty shops
        if shop_entry["categories"]:
            shops_result.append(shop_entry)

    embedding_cache_full["shops"] = shops_result
    embedding_cache_full["total_shops"] = len(shops_result)
    embedding_cache_full["last_updated"] = time.time()

    print(f"[READY] Cached {len(shops_result)} shops in {round((time.time()-start)*1000,2)}ms")


def on_full_item_snapshot(col_snapshot, changes, read_time):
    print("[LISTENER] Firestore change → refreshing FULL cache")
    refresh_full_item_cache()


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
# SALES SCAN (SHOP-STRICT)
# ======================================================
@app.route("/sales", methods=["POST"])
def sales_scan():
    try:
        data = request.get_json(force=True)
        print("📥 /sales HIT")

        shop_id = data.get("shop_id")
        frame_b64 = data.get("frame")

        if not shop_id or not frame_b64:
            print("❌ Missing shop_id or frame")
            return jsonify({"match": None})

        # decode image
        if frame_b64.startswith("data:image"):
            frame_b64 = frame_b64.split(",")[1]

        img = Image.open(BytesIO(base64.b64decode(frame_b64))).convert("RGB")
        frame = np.array(img)
        query_embedding = generate_embedding(frame).reshape(1, -1)

        # 🔐 HARD shop isolation
        shop = next(
            (s for s in embedding_cache_full["shops"] if s["shop_id"] == shop_id),
            None
        )

        if not shop:
            print("⛔ Shop NOT found in cache")
            return jsonify({"match": None})

        best_match = None
        best_score = 0.0

        for category in shop["categories"]:
            for item in category["items"]:
                for vector in item["embeddings"]:
                    score = cosine_similarity(
                        query_embedding,
                        vector.reshape(1, -1)
                    )[0][0]

                    if score > best_score:
                        best_score = score
                        best_match = {
                            "item_id": item["item_id"],
                            "category_id": category["category_id"],  # ✅ ADDED
                            "name": item["name"],
                            "thumbnail": item["thumbnail"],
                            "sellPrice": item.get("sellPrice", 0),  # ✅ ADDED
                            "score": round(float(score), 3)
                        }

        print("🎯 Best score:", best_score)

        # 🔐 STRICT threshold
        if not best_match or best_score < 0.5:
            print("❌ No valid match in THIS shop")
            return jsonify({"match": None})

        print("✅ Match confirmed:", best_match)
        return jsonify({"match": best_match})

    except Exception as e:
        print("🔥 /sales error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"match": None}), 500
# ======================================================
# COMPLETE SALES (UPDATED WITH SELLING UNITS SUPPORT)
# ======================================================
@app.route("/complete-sale", methods=["POST"])
def complete_sale():
    """
    UPDATED: Now handles both main items AND selling units
    """
    try:
        start_time = time.time()
        data = request.get_json() or {}

        shop_id = (data.get("shop_id") or "").strip()
        items = data.get("items", [])
        payment = data.get("payment", {}) or {}
        user_id = data.get("user_id")  # Firebase auth uid (owner or staff)
        seller = data.get("seller") or {}
        session_type = data.get("sessionType") or seller.get("type") or "owner"

        # Validation
        if not shop_id:
            return jsonify({"success": False, "error": "shop_id is required"}), 400

        if not items or not isinstance(items, list):
            return jsonify({"success": False, "error": "items array is required"}), 400

        # Receipt ID
        timestamp_ms = int(time.time() * 1000)
        receipt_id = f"RCPT-{timestamp_ms}-{random.randint(1000, 9999)}"

        # PerformedBy tracking
        performed_by = {
            "type": seller.get("type", "owner"),
            "authUid": seller.get("authUid") or user_id,
            "staffId": seller.get("staffId"),
            "name": seller.get("name"),
            "email": seller.get("email"),
            "roleName": seller.get("roleName"),
            "accessLevel": seller.get("accessLevel"),
        }

        added_by_display = (
            performed_by.get("name")
            or performed_by.get("email")
            or f"sale_{receipt_id}"
        )

        print(f"✅ Processing sale: {receipt_id}")
        print(f"   Shop: {shop_id}")
        print(f"   Performed by: {performed_by.get('type')} | {added_by_display}")
        print(f"   Items to process: {len(items)}")

        updated_items = []

        # Process each item in the cart
        for cart_item in items:
            item_type = cart_item.get("type", "main_item")
            item_id = cart_item.get("item_id") or cart_item.get("id")
            category_id = cart_item.get("category_id", "unknown")
            sold_qty = float(cart_item.get("quantity", 1) or 1)
            
            if not item_id:
                print(f"⚠️ Item missing ID: {cart_item}")
                continue

            try:
                item_ref = (
                    db.collection("Shops").document(shop_id)
                    .collection("categories").document(category_id)
                    .collection("items").document(item_id)
                )

                item_doc = item_ref.get()
                if not item_doc.exists:
                    print(f"⚠️ Item {item_id} not found in shop {shop_id}")
                    continue

                item_data = item_doc.to_dict() or {}
                current_stock = float(item_data.get("stock", 0) or 0)
                
                # Get sell price from cart_item (use camelCase with fallback to snake_case)
                sell_price = cart_item.get("sellPrice") or cart_item.get("sell_price") or 0
                
                # Calculate stock deduction based on item type
                if item_type == "selling_unit":
                    # For selling units: get conversion factor
                    sell_unit_id = cart_item.get("sell_unit_id")
                    conversion_factor = float(cart_item.get("conversion_factor", 1.0))
                    
                    # Calculate main stock deduction
                    main_stock_deduction = sold_qty / conversion_factor
                    
                    # Check if enough stock
                    if current_stock < main_stock_deduction:
                        return jsonify({
                            "success": False,
                            "error": f"Insufficient stock for {cart_item.get('display_name', cart_item.get('name','item'))}. "
                                     f"Available: {round(current_stock * conversion_factor, 2)} units, Requested: {sold_qty}"
                        }), 400
                    
                    new_stock = current_stock - main_stock_deduction
                    
                    # Record selling unit details
                    unit_name = cart_item.get("display_name", "Unit")
                    
                else:  # main_item
                    # For main items: direct deduction
                    if current_stock < sold_qty:
                        return jsonify({
                            "success": False,
                            "error": f"Insufficient stock for {cart_item.get('name','item')}. "
                                     f"Available: {current_stock}, Requested: {sold_qty}"
                        }), 400
                    
                    new_stock = current_stock - sold_qty
                    main_stock_deduction = sold_qty
                    unit_name = item_data.get("name", "Item")
                    conversion_factor = 1.0

                # Update stock in Firestore
                item_ref.update({
                    "stock": new_stock,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "lastTransactionId": f"sale_{receipt_id}",
                    "lastStockUpdate": firestore.SERVER_TIMESTAMP
                })

                # Create stock transaction record
                stock_transaction = {
                    "id": f"sale_{receipt_id}_{item_id}_{cart_item.get('sell_unit_id', '')}",
                    "quantity": -main_stock_deduction,  # In main units (negative for sale)
                    "date": datetime.now().strftime("%d/%m/%Y"),
                    "timestamp": timestamp_ms,
                    "type": "sale",
                    "receipt_id": receipt_id,
                    "price": sell_price,  # Use the sell_price variable
                    
                    # For selling units: add unit info
                    "unit_type": item_type,
                    "unit_name": unit_name,
                    "conversion_factor": conversion_factor if item_type == "selling_unit" else 1.0,
                    "sold_units": sold_qty if item_type == "selling_unit" else None,
                    
                    # Tracking
                    "addedBy": added_by_display,
                    "performedBy": performed_by,
                    "sessionType": session_type
                }

                item_ref.update({
                    "stockTransactions": firestore.ArrayUnion([stock_transaction])
                })

                updated_items.append({
                    "item_id": item_id,
                    "sell_unit_id": cart_item.get("sell_unit_id") if item_type == "selling_unit" else None,
                    "name": cart_item.get("name"),
                    "display_name": unit_name,
                    "type": item_type,
                    "sold_quantity": sold_qty,
                    "main_stock_deduction": main_stock_deduction,
                    "remaining_stock": new_stock,
                    "price": sell_price,  # Use the sell_price variable
                    "conversion_factor": conversion_factor if item_type == "selling_unit" else None
                })

                print(f"   📦 Updated {item_id} ({item_type}): {current_stock} → {new_stock} (-{main_stock_deduction}) Price: ${sell_price}")

            except Exception as e:
                print(f"❌ Error updating item {item_id}: {e}")
                import traceback
                traceback.print_exc()

        # Save receipt
        receipt_data = {
            "receipt_id": receipt_id,
            "shop_id": shop_id,
            "user_id": user_id,
            "items": items,
            "payment": payment,
            "total": payment.get("total", 0),
            "updated_items": updated_items,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "created_at": firestore.SERVER_TIMESTAMP,
            "status": "completed",
            "archived": False,

            # Tracking
            "performedBy": performed_by,
            "addedBy": added_by_display,
            "sessionType": session_type
        }

        # Save to global receipts
        db.collection("receipts").document(receipt_id).set(receipt_data)
        # Save to shop receipts
        db.collection("Shops").document(shop_id).collection("receipts").document(receipt_id).set(receipt_data)

        return jsonify({
            "success": True,
            "receipt_id": receipt_id,
            "updated_items": updated_items,
            "performedBy": performed_by,
            "addedBy": added_by_display,
            "message": f"Sale completed. {len(updated_items)} items updated",
            "items_count": len(items),
            "total": payment.get("total", 0),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        })

    except Exception as e:
        print(f"🔥 /complete-sale error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ======================================================
# ITEM OPTIMIZATION
# ======================================================
@app.route("/item-optimization", methods=["GET"])
def item_optimization():
    return jsonify({
        "status": "success",
        "shops": embedding_cache_full["shops"],
        "total_shops": embedding_cache_full["total_shops"],
        "last_updated": embedding_cache_full["last_updated"]
    })


# ======================================================
# DEBUG ENDPOINT (OPTIONAL)
# ======================================================
@app.route("/debug-cache", methods=["GET"])
def debug_cache():
    """Debug endpoint to check cache contents"""
    if not embedding_cache_full["shops"]:
        return jsonify({"error": "Cache empty"}), 404
    
    # Check first shop, first category, first item
    try:
        first_shop = embedding_cache_full["shops"][0]
        first_category = first_shop["categories"][0]
        first_item = first_category["items"][0]
        
        return jsonify({
            "has_sellPrice": "sellPrice" in first_item,
            "sellPrice_value": first_item.get("sellPrice"),
            "item_name": first_item["name"],
            "cache_details": {
                "total_shops": len(embedding_cache_full["shops"]),
                "last_updated": embedding_cache_full["last_updated"]
            }
        })
    except (IndexError, KeyError) as e:
        return jsonify({"error": f"Cache structure issue: {str(e)}"}), 500


# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    print("[INIT] Preloading FULL cache...")
    refresh_full_item_cache()
    db.collection_group("items").on_snapshot(on_full_item_snapshot)

    app.run(debug=True)