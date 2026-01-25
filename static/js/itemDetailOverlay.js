// itemDetailOverlay.js - UPDATED WITH BATCH TRACKING + UNIT MANAGEMENT + SELL UNIT PRICING
// ✅ Preserves existing functionality
// ✅ Captures baseUnit if missing
// ✅ Batch-aware stock intake
// ✅ Saves images to Cloudinary (unsigned)
// ✅ Sets overlay dataset for other modules (selling units)
// ✅ Emits lifecycle events (opened/base-unit-set/closed)
// ✅ Exposes getCurrentItemContext()
// ✅ Prompts per-selling-unit price at every stock intake (create/missing/optional update)

import { db } from "./firebase-config.js";
import {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  arrayUnion,
  increment,
  collection,
  getDocs,
  addDoc,
  deleteDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const CLOUDINARY_CLOUD = "decckqobb";
const CLOUDINARY_UPLOAD_PRESET = "Superkeeper";
const FLASK_BACKEND_URL = window.location.origin;

document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.getElementById("overlay");
  const overlayContent = document.querySelector(".overlay-content");
  const itemDetail = document.getElementById("item-detail");
  const itemNameEl = document.getElementById("item-name");
  const itemMeta = document.getElementById("item-meta");
  const editToggleBtn = document.getElementById("edit-toggle-btn");

  let currentItem = null;
  let captureInProgress = false;
  let capturePhase = null;
  let editMode = false;

  console.log("itemDetailOverlay.js loaded (batch tracking + sell unit pricing)");

  // One-time: wire Selling Units module if present
  try {
    window.sellingUnitsConfigure?.({
      db,
      auth: getAuth(),
      FS: { collection, addDoc, getDocs, deleteDoc, doc, updateDoc, serverTimestamp },
      cloudName: CLOUDINARY_CLOUD,
      uploadPreset: CLOUDINARY_UPLOAD_PRESET
    });
  } catch (e) {
    console.warn("[SellingUnits] configure skipped:", e);
  }

  // =========================================================
  // SESSION / RBAC / ACTOR HELPERS
  // =========================================================
  function safeParse(json) {
    try { return JSON.parse(json); } catch { return null; }
  }

  function getSessionType() {
    return localStorage.getItem("sessionType") || "owner";
  }

  function getAccessLevel() {
    if (getSessionType() === "staff") {
      const ctx = safeParse(localStorage.getItem("staffContext")) || {};
      return Number(ctx.accessLevel ?? 1);
    }
    return 4; // owner
  }

  function canManageStock() {
    return getAccessLevel() >= 2;
  }

  function getActiveShopId() {
    return localStorage.getItem("activeShopId");
  }

  function getActor() {
    const auth = getAuth();
    const user = auth.currentUser;
    const sessionType = getSessionType();

    if (sessionType === "staff") {
      const ctx = safeParse(localStorage.getItem("staffContext")) || {};
      return {
        type: "staff",
        authUid: user?.uid || null,
        staffId: ctx.staffId || null,
        name: ctx.name || user?.displayName || "",
        email: ctx.email || user?.email || "",
        roleName: ctx.roleName || "",
        accessLevel: ctx.accessLevel ?? null,
        shopId: ctx.shopId || getActiveShopId() || null
      };
    }

    return {
      type: "owner",
      authUid: user?.uid || null,
      name: user?.displayName || "",
      email: user?.email || "",
      accessLevel: 4,
      shopId: getActiveShopId() || user?.uid || null
    };
  }

  function actorDisplayName(actor) {
    return actor?.name || actor?.email || actor?.authUid || "User";
  }

  function syncEditButtonUI() {
    if (!editToggleBtn) return;

    if (canManageStock()) {
      editToggleBtn.disabled = false;
      editToggleBtn.style.opacity = "1";
      editToggleBtn.title = "";
      return;
    }

    editToggleBtn.disabled = true;
    editToggleBtn.style.opacity = "0.6";
    editToggleBtn.title = "Access denied: you cannot edit items/stock.";
    editToggleBtn.textContent = "Edit";
    editMode = false;
  }

  // =========================================================
  // UNIT & BATCH HELPERS
  // =========================================================
  async function getBaseUnitFromUser(itemName) {
    const exampleText =
      `How do you measure "${itemName}"?\n\n` +
      `Examples:\n` +
      `• "bag" for sugar (you buy in bags)\n` +
      `• "kg" for rice (you buy by weight)\n` +
      `• "carton" for soap (comes in cartons)\n` +
      `• "piece" for items you count\n\n` +
      `Enter your measurement unit:`;

    const unit = prompt(exampleText, "piece");
    return unit ? unit.trim().toLowerCase() : "piece";
  }

  async function getBatchPrices(itemData, isFirstBatch = false) {
    const unit = itemData.baseUnit || "unit";
    const currentBuy = itemData.buyPrice || 0;
    const currentSell = itemData.sellPrice || 0;

    if (isFirstBatch) {
      const buyMsg = `Buying price per ${unit}?\nExample: $30 per bag`;
      const buyPrice = parseFloat(prompt(buyMsg, currentBuy || ""));
      if (isNaN(buyPrice)) return null;

      const sellMsg = `Selling price per ${unit}?\nExample: $35 per bag`;
      const sellPrice = parseFloat(prompt(sellMsg, currentSell || ""));
      if (isNaN(sellPrice)) return null;

      return { buyPrice, sellPrice, updateBuy: true, updateSell: true };
    }

    const msg =
      `Adding more stock of "${itemData.name}"\n\n` +
      `Current prices (per ${unit}):\n` +
      `Buy: $${currentBuy} | Sell: $${currentSell}\n\n` +
      `Have prices changed?\n` +
      `• Enter "b" to update buy price only\n` +
      `• Enter "s" to update sell price only\n` +
      `• Enter "bs" to update both\n` +
      `• Enter "n" for no change\n\n` +
      `Enter your choice (b/s/bs/n):`;

    const choice = prompt(msg, "n")?.toLowerCase() || "n";

    let buyPrice = currentBuy;
    let sellPrice = currentSell;
    let updateBuy = false;
    let updateSell = false;

    if (choice.includes("b")) {
      const newBuy = parseFloat(prompt(`New buying price per ${unit}:`, currentBuy));
      if (!isNaN(newBuy)) {
        buyPrice = newBuy;
        updateBuy = true;
      }
    }

    if (choice.includes("s")) {
      const newSell = parseFloat(prompt(`New selling price per ${unit}:`, currentSell));
      if (!isNaN(newSell)) {
        sellPrice = newSell;
        updateSell = true;
      }
    }

    return { buyPrice, sellPrice, updateBuy, updateSell };
  }

  function calculateFIFOCost(batches, quantity) {
    if (!batches || batches.length === 0) return 0;

    let remaining = quantity;
    let totalCost = 0;
    const sortedBatches = [...batches].sort((a, b) => a.timestamp - b.timestamp);

    for (const batch of sortedBatches) {
      if (remaining <= 0) break;
      const available = batch.quantity || 0;
      const use = Math.min(available, remaining);
      totalCost += use * (batch.buyPrice || 0);
      remaining -= use;
    }

    return totalCost;
  }

  // =========================================================
  // SELL-UNITS PRICING HELPERS
  // =========================================================
  // Fetch selling units from sellUnits and sellingUnits (supports both, returns combined list)
  async function fetchSellUnits(shopId, categoryId, itemId) {
    const colNames = ["sellUnits", "sellingUnits"];
    const results = [];
    for (const colName of colNames) {
      try {
        const colRef = collection(db, "Shops", shopId, "categories", categoryId, "items", itemId, colName);
        const snap = await getDocs(colRef);
        snap.docs.forEach(d => results.push({ id: d.id, __col: colName, ...d.data() }));
      } catch (e) {
        console.warn(`fetchSellUnits: failed for ${colName}`, e);
      }
    }
    return results;
  }

  // Prompt for prices per unit and update
  // Rules:
  // - If a unit has no sellPrice: require a price (with suggestion).
  // - If a unit has sellPrice: ask if user wants to change; if yes, prompt for new price (with suggestion).
  async function ensureSellUnitPrices(ctx, baseSellPrice, actor) {
    try {
      const units = await fetchSellUnits(ctx.shopId, ctx.categoryId, ctx.itemId);
      if (!units || units.length === 0) return;

      const timestamp = Date.now();
      for (const u of units) {
        const conv = Number(u.conversionFactor ?? u.conversion ?? 1) || 1;
        const suggested = Math.round(((Number(baseSellPrice || 0) / conv) + Number.EPSILON) * 100) / 100;

        // No price yet -> must capture
        if (u.sellPrice == null) {
          let val = prompt(
            `Set selling price for "${u.name}" (no price yet)\n` +
            `Suggestion: ${suggested} (main price ÷ ${conv})\n\n` +
            `Enter price per ${u.name}:`,
            String(suggested)
          );
          if (val === null) continue; // allow skipping
          const price = parseFloat(val);
          if (!isFinite(price) || price <= 0) {
            alert(`Skipping "${u.name}" — invalid price.`);
            continue;
          }
          const uRef = doc(db, "Shops", ctx.shopId, "categories", ctx.categoryId, "items", ctx.itemId, u.__col || "sellUnits", u.id);
          await updateDoc(uRef, {
            sellPrice: price,
            lastSellPriceUpdate: timestamp,
            updatedAt: timestamp,
            updatedBy: actor
          });
          continue;
        }

        // Has price -> ask if change
        const wantsChange = confirm(
          `"${u.name}" current price: ${u.sellPrice}\n` +
          `Suggested (main price ÷ ${conv}): ${suggested}\n\n` +
          `Do you want to change it now?`
        );
        if (!wantsChange) continue;

        let val = prompt(
          `Enter new price for "${u.name}":\n` +
          `Suggestion: ${suggested}`,
          String(u.sellPrice)
        );
        if (val === null) continue;
        const price = parseFloat(val);
        if (!isFinite(price) || price <= 0) {
          alert(`Skipping "${u.name}" — invalid price.`);
          continue;
        }
        const uRef = doc(db, "Shops", ctx.shopId, "categories", ctx.categoryId, "items", ctx.itemId, u.__col || "sellUnits", u.id);
        await updateDoc(uRef, {
          sellPrice: price,
          lastSellPriceUpdate: timestamp,
          updatedAt: timestamp,
          updatedBy: actor
        });
      }
    } catch (e) {
      console.warn("ensureSellUnitPrices failed:", e);
    }
  }

  // =========================================================
  // STYLES
  // =========================================================
  function injectAlertStyles() {
    if (!document.getElementById("alert-styles")) {
      const style = document.createElement("style");
      style.id = "alert-styles";
      style.textContent = `
        /* Existing styles remain unchanged */
        #item-detail .item-images {
          display: flex !important;
          flex-direction: row !important;
          flex-wrap: nowrap !important;
          gap: 15px !important;
          margin: 20px 0 !important;
          justify-content: space-between !important;
          width: 100% !important;
        }
        #item-detail .image-slot {
          flex: 1 !important;
          width: 48% !important;
          min-width: 48% !important;
          max-width: 48% !important;
          height: 200px !important;
          border: 2px dashed #ccc !important;
          border-radius: 10px !important;
          overflow: hidden !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          background: #f9f9f9 !important;
          position: relative !important;
          float: left !important;
          box-sizing: border-box !important;
        }
        #item-detail .image-slot.placeholder { color: #999; font-style: italic; font-size: 14px; padding: 20px; text-align: center; }
        #item-detail .item-thumb { width: 100% !important; height: 100% !important; object-fit: cover !important; display: block !important; }
        #item-detail .image-edit-overlay {
          position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0, 0, 0, 0.7);
          padding: 8px; display: flex; justify-content: center; gap: 10px; opacity: 0; transition: opacity 0.3s ease;
        }
        #item-detail .image-slot:hover .image-edit-overlay { opacity: 1; }
        #item-detail .retake-btn { background: #ff6b6b; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        #item-detail .pencil { color: white; }

        .alert-slider-container { margin: 20px 0; padding: 15px; background: #f0fff0; border: 2px solid #2ed573; border-radius: 10px; transition: all 0.3s ease; clear: both; }
        .alert-slider-container.alert-active { background: #fff5f5; border-color: #ff6b6b; }
        .alert-slider-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .alert-title { font-size: 16px; font-weight: bold; color: #2ed573; }
        .alert-title.alert-active { color: #ff6b6b; }
        .alert-status { font-size: 12px; padding: 4px 8px; border-radius: 12px; background: #2ed57320; color: #2ed573; font-weight: 500; }
        .alert-status.alert-active { background: #ff6b6b20; color: #ff6b6b; }
        .alert-config-panel { display: none; margin-top: 15px; padding: 20px; background: white; border-radius: 8px; border: 1px solid #e9ecef; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }

        .alert-slider { width: 100%; height: 8px; -webkit-appearance: none; background: linear-gradient(to right, #2ed573, #ff6b6b); border-radius: 4px; outline: none; margin: 10px 0; }
        .alert-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 24px; height: 24px; border-radius: 50%; background: #667eea; cursor: pointer; border: 3px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }

        .alert-value-display { font-size: 18px; font-weight: bold; color: #667eea; text-align: center; margin: 10px 0; }
        .alert-labels { display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #666; }
        .alert-buttons { display: flex; gap: 10px; margin-top: 15px; }
        .alert-save-btn { flex: 1; padding: 12px; background: linear-gradient(135deg, #2ed573, #1dd1a1); color: white; border: none; border-radius: 6px; font-weight: 600; font-size: 14px; cursor: pointer; }
        .alert-cancel-btn { flex: 1; padding: 12px; background: #f8f9fa; color: #666; border: 2px solid #e9ecef; border-radius: 6px; font-weight: 600; font-size: 14px; cursor: pointer; }

        #item-detail .item-prices { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; clear: both; }
        #item-detail .capture-actions { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; text-align: center; }
        #item-detail .capture-status { margin-top: 10px; color: #666; font-style: italic; text-align: center; }

        #item-detail .image-slot:nth-child(1),
        #item-detail .image-slot:nth-child(2) { float: left !important; display: block !important; width: 48% !important; margin-right: 2% !important; }
        #item-detail .image-slot:nth-child(2) { margin-right: 0 !important; }

        @media (min-width: 480px) {
          #item-detail .item-images { display: block !important; font-size: 0 !important; }
          #item-detail .image-slot { display: inline-block !important; vertical-align: top !important; width: 49% !important; margin-right: 2% !important; }
          #item-detail .image-slot:last-child { margin-right: 0 !important; }
        }
      `;
      document.head.appendChild(style);
    }
  }

  // =========================================================
  // ALERT HELPERS
  // =========================================================
  function updateAlertSliderValue(sliderId, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (slider && display) {
      const value = parseInt(slider.value);
      display.textContent = value;
      const percent = (value / 50) * 100;
      slider.style.background = `linear-gradient(to right, #2ed573 ${percent}%, #ff6b6b ${percent}%)`;
      return value;
    }
    return 0;
  }

  async function saveAlertSettings() {
    if (!currentItem) return;
    if (!canManageStock()) {
      alert("Access denied: You don't have permission to change alert settings.");
      return;
    }

    const actor = getActor();
    const alertValue = parseInt(document.getElementById("alert-slider")?.value || 5);
    const alertDescription = document.getElementById("alert-description")?.value.trim() || "";

    const itemRef = doc(
      db, "Shops", currentItem.uid,
      "categories", currentItem.categoryId,
      "items", currentItem.itemId
    );

    try {
      await updateDoc(itemRef, {
        lowStockAlert: alertValue,
        alertDescription: alertDescription || null,
        alertLastChecked: Date.now(),
        updatedAt: Date.now(),
        updatedBy: actor
      });

      currentItem.data.lowStockAlert = alertValue;
      currentItem.data.alertDescription = alertDescription || null;
      renderItemMeta(currentItem.data);
    } catch (error) {
      console.error("❌ Error saving alert:", error);
      alert("Failed to save alert settings. Please try again.");
    }
  }

  function getBufferStatus(stock, threshold) {
    const buffer = stock - threshold;
    if (buffer > 10) return { class: "healthy", text: "Healthy Buffer" };
    if (buffer > 0) return { class: "warning", text: "Low Buffer" };
    return { class: "critical", text: "Needs Attention" };
  }

  // =========================================================
  // BACKEND NOTIFY (UNCHANGED)
  // =========================================================
  function sendImageForEmbedding(imageUrl, imageIndex) {
    if (!imageUrl || !currentItem || imageIndex == null) return;

    const payload = {
      event: "image_saved",
      image_url: imageUrl,
      item_id: currentItem.itemId,
      shop_id: currentItem.uid,
      category_id: currentItem.categoryId,
      image_index: imageIndex,
      timestamp: Date.now()
    };

    fetch(`${FLASK_BACKEND_URL}/vectorize-item`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => console.log("[BACKEND RESPONSE]", data))
      .catch(err => console.warn("Backend request failed:", err));
  }

  // =========================================================
  // EDIT/SAVE TOGGLE
  // =========================================================
  if (editToggleBtn) {
    editToggleBtn.addEventListener("click", async () => {
      if (!canManageStock()) {
        alert("Access denied: You don't have permission to edit items.");
        return;
      }
      if (!currentItem) return;

      if (editMode) {
        await saveEdits();
        editMode = false;
        editToggleBtn.textContent = "Edit";
      } else {
        editMode = true;
        editToggleBtn.textContent = "Save";
      }
      renderItemMeta(currentItem?.data);
    });
  }

  async function saveEdits() {
    if (!currentItem) return;
    if (!canManageStock()) return;

    const actor = getActor();
    const nameInput = document.getElementById("item-name-input");
    const buyPriceInput = document.getElementById("buy-price-input");
    const sellPriceInput = document.getElementById("sell-price-input");

    const updatedName = nameInput ? nameInput.value.trim() : currentItem.data.name;
    const updatedBuyPrice = buyPriceInput ? parseFloat(buyPriceInput.value) || 0 : currentItem.data.buyPrice || 0;
    const updatedSellPrice = sellPriceInput ? parseFloat(sellPriceInput.value) || 0 : currentItem.data.sellPrice || 0;

    const itemRef = doc(
      db,
      "Shops", currentItem.uid,
      "categories", currentItem.categoryId,
      "items", currentItem.itemId
    );

    try {
      await updateDoc(itemRef, {
        name: updatedName,
        buyPrice: updatedBuyPrice,
        sellPrice: updatedSellPrice,
        updatedAt: Date.now(),
        updatedBy: actor
      });

      currentItem.data.name = updatedName;
      currentItem.data.buyPrice = updatedBuyPrice;
      currentItem.data.sellPrice = updatedSellPrice;
      currentItem.name = updatedName;

      // Keep dataset itemName fresh
      itemDetail.dataset.itemName = updatedName;
    } catch (error) {
      console.error("Error updating item:", error);
      alert("Failed to save changes. Please try again.");
    }
  }

  // =========================================================
  // SHOW ITEM DETAIL (sets dataset)
  // =========================================================
  async function showItemDetail(name, uid, categoryId, itemId) {
    editMode = false;
    if (editToggleBtn) editToggleBtn.textContent = "Edit";

    injectAlertStyles();

    const shopId = getActiveShopId();
    if (!shopId) {
      alert("Missing shop context. Please login again.");
      window.location.href = "/";
      return;
    }

    uid = shopId;

    overlay.classList.remove("hidden");
    overlayContent.classList.add("hidden");
    itemDetail.classList.remove("hidden");
    document.body.style.overflow = "hidden";

    const itemRef = doc(db, "Shops", uid, "categories", categoryId, "items", itemId);
    const snap = await getDoc(itemRef);

    const data = snap.exists()
      ? snap.data()
      : {
          name,
          images: [],
          stock: 0,
          stockTransactions: [],
          batches: [],
          lowStockAlert: 5,
          alertDescription: ""
        };

    currentItem = { uid, categoryId, itemId, name, data };

    // Attach context to overlay dataset
    itemDetail.dataset.shopId = uid;
    itemDetail.dataset.categoryId = categoryId;
    itemDetail.dataset.itemId = itemId;
    itemDetail.dataset.itemName = data.name || name || "";
    if (data.baseUnit) itemDetail.dataset.baseUnit = data.baseUnit;

    document.dispatchEvent(new CustomEvent("item-detail:opened", {
      detail: {
        shopId: uid,
        categoryId,
        itemId,
        itemName: data.name || name || "",
        baseUnit: data.baseUnit || null
      }
    }));

    syncEditButtonUI();
    renderItemMeta(data);
    injectItemDetailCloseButton();

    if ((!Array.isArray(data.images) || data.images.length === 0) && canManageStock()) {
      await captureImage1(itemRef, data);
      return;
    }

    if (Array.isArray(data.images) && data.images.length === 1 && canManageStock()) {
      capturePhase = "awaiting-image-2";
      injectCaptureCTA(itemRef, data.images);
      return;
    }

    if ((data.buyPrice == null || data.sellPrice == null) && canManageStock()) {
      await ensurePrices(itemRef, data);
      renderItemMeta(currentItem.data);
    }
  }

  function injectItemDetailCloseButton() {
    if (itemDetail && !document.getElementById("item-detail-close-btn")) {
      const closeBtn = document.createElement("span");
      closeBtn.id = "item-detail-close-btn";
      closeBtn.className = "close-x";
      closeBtn.setAttribute("role", "button");
      closeBtn.setAttribute("aria-label", "Close item detail and go back to categories");
      closeBtn.innerHTML = "&times;";
      closeBtn.style.position = "absolute";
      closeBtn.style.top = "10px";
      closeBtn.style.right = "15px";
      closeBtn.style.fontSize = "24px";
      closeBtn.style.cursor = "pointer";
      closeBtn.style.zIndex = "1002";
      itemDetail.appendChild(closeBtn);

      closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (captureInProgress) return alert("Finish image capture first.");
        hideItemDetail();
      });
    }
  }

  // =========================================================
  // CAPTURE FLOW
  // =========================================================
  async function captureImage1(itemRef, data) {
    if (!canManageStock()) return;
    const actor = getActor();
    capturePhase = "processing-image-1";
    showStatus("Opening camera for photo 1…");
    setPlaceholderMessage(0, "Processing first image…");

    const file = await promptCameraCapture();
    clearStatus();
    if (!file) {
      capturePhase = null;
      setPlaceholderMessage(0, "No image");
      return;
    }

    const preview = URL.createObjectURL(file);
    setPreviewImageSlot(0, preview);
    const url = await uploadToCloudinary(file);
    const images = [url];

    await setDoc(
      itemRef,
      {
        ...data,
        images,
        stock: data.stock ?? 0,
        stockTransactions: Array.isArray(data.stockTransactions) ? data.stockTransactions : [],
        batches: Array.isArray(data.batches) ? data.batches : [],
        lowStockAlert: data.lowStockAlert ?? 5,
        createdAt: data.createdAt ?? Date.now(),
        createdBy: data.createdBy ?? actor,
        updatedAt: Date.now(),
        updatedBy: actor
      },
      { merge: true }
    );

    currentItem.data.images = images;
    setPreviewImageSlot(0, url);
    sendImageForEmbedding(url, 0);
    capturePhase = "awaiting-image-2";
    injectCaptureCTA(itemRef, images);
  }

  function injectCaptureCTA(itemRef, images) {
    if (!canManageStock()) return;
    const actions = itemMeta.querySelector(".capture-actions");
    if (!actions) return;
    actions.innerHTML = "";

    const info = document.createElement("div");
    info.textContent = "Photo 1 saved. Capture image 2.";
    info.style.marginBottom = "6px";

    const btn = document.createElement("button");
    btn.textContent = "Capture image 2";
    btn.style.padding = "10px 14px";

    btn.onclick = async () => {
      if (!canManageStock()) return;
      const actor = getActor();
      btn.disabled = true;
      capturePhase = "processing-image-2";
      info.textContent = "Opening camera for photo 2…";
      showStatus("Preparing second image…");
      setPlaceholderMessage(1, "Processing image 2…");

      const file = await promptCameraCapture();
      clearStatus();
      if (!file) {
        btn.disabled = false;
        capturePhase = "awaiting-image-2";
        info.textContent = "Photo 1 saved. Capture image 2.";
        setPlaceholderMessage(1, "No image");
        return;
      }

      const preview = URL.createObjectURL(file);
      setPreviewImageSlot(1, preview);
      showStatus("Uploading second image…");
      const url = await uploadToCloudinary(file);
      images[1] = url;

      await updateDoc(itemRef, { images, updatedAt: Date.now(), updatedBy: actor });
      currentItem.data.images = images;
      setPreviewImageSlot(1, url);
      sendImageForEmbedding(url, 1);
      capturePhase = null;
      actions.innerHTML = "";
      showStatus("Image 2 saved. Processing item…");

      await ensurePrices(itemRef, currentItem.data);
      renderItemMeta(currentItem.data);
      clearStatus();
    };

    actions.append(info, btn);
  }

  // =========================================================
  // ensurePrices with baseUnit + dataset sync
  // =========================================================
  async function ensurePrices(itemRef, data) {
    if (!canManageStock()) return;
    if (data.buyPrice != null && data.sellPrice != null && data.baseUnit) return;

    const actor = getActor();

    if (!data.baseUnit) {
      const baseUnit = await getBaseUnitFromUser(data.name);
      if (!baseUnit) return;

      await updateDoc(itemRef, {
        baseUnit: baseUnit,
        updatedAt: Date.now(),
        updatedBy: actor
      });
      data.baseUnit = baseUnit;

      itemDetail.dataset.baseUnit = baseUnit;
      document.dispatchEvent(new CustomEvent("item-detail:base-unit-set", {
        detail: {
          shopId: currentItem?.uid,
          categoryId: currentItem?.categoryId,
          itemId: currentItem?.itemId,
          baseUnit
        }
      }));
    }

    if (data.buyPrice == null || data.sellPrice == null) {
      const priceResult = await getBatchPrices(data, true);
      if (!priceResult) return;

      await updateDoc(itemRef, {
        buyPrice: priceResult.buyPrice,
        sellPrice: priceResult.sellPrice,
        updatedAt: Date.now(),
        updatedBy: actor
      });

      data.buyPrice = priceResult.buyPrice;
      data.sellPrice = priceResult.sellPrice;
    }
  }

  // =========================================================
  // addStockToItem with batch tracking + SELL UNIT PRICING
  // =========================================================
  async function addStockToItem() {
    if (!currentItem) return;
    if (!canManageStock()) {
      alert("Access denied: You don't have permission to add stock.");
      return;
    }

    const actor = getActor();
    const itemData = currentItem.data;
    const baseUnit = itemData.baseUnit || "unit";

    const quantity = parseFloat(prompt(
      `How many ${baseUnit} of "${currentItem.name}" to add?\n\n` +
      `Example: If adding 3 bags, enter "3"\n` +
      `Current stock: ${itemData.stock || 0} ${baseUnit}`,
      "1"
    ));

    if (isNaN(quantity) || quantity <= 0) {
      alert("Please enter a valid number");
      return;
    }

    const timestamp = Date.now();
    const date = new Date().toLocaleDateString();

    const batches = itemData.batches || [];
    const isFirstBatch = batches.length === 0;

    const priceResult = await getBatchPrices(itemData, isFirstBatch);
    if (!priceResult) {
      alert("Stock addition cancelled.");
      return;
    }

    const batchId = `batch_${timestamp}_${Math.random().toString(36).substr(2, 6)}`;
    const batch = {
      id: batchId,
      quantity: quantity,
      unit: baseUnit,
      buyPrice: priceResult.buyPrice,
      sellPrice: priceResult.sellPrice,
      date: date,
      timestamp: timestamp,
      addedBy: actorDisplayName(actor),
      performedBy: actor
    };

    const txnId = `stock_${timestamp}_${Math.random().toString(36).substr(2, 6)}`;
    const stockTransaction = {
      id: txnId,
      quantity: quantity,
      date: date,
      timestamp: timestamp,
      type: "stock_in",
      batchId: batchId,
      unit: baseUnit,
      buyPrice: priceResult.buyPrice,
      sellPrice: priceResult.sellPrice,
      addedBy: actorDisplayName(actor),
      performedBy: actor,
      sessionType: getSessionType()
    };

    const itemRef = doc(
      db, "Shops", currentItem.uid,
      "categories", currentItem.categoryId,
      "items", currentItem.itemId
    );

    try {
      const updates = {
        stockTransactions: arrayUnion(stockTransaction),
        stock: increment(quantity),
        batches: arrayUnion(batch),
        lastTransactionId: txnId,
        lastStockUpdate: timestamp,
        updatedAt: timestamp,
        updatedBy: actor
      };

      if (priceResult.updateBuy) {
        updates.buyPrice = priceResult.buyPrice;
        updates.lastBuyPriceUpdate = timestamp;
      }
      if (priceResult.updateSell) {
        updates.sellPrice = priceResult.sellPrice;
        updates.lastSellPriceUpdate = timestamp;
      }
      if (!itemData.baseUnit) {
        updates.baseUnit = baseUnit;
      }

      // 1) Update the main item with batch and stock changes
      await updateDoc(itemRef, updates);

      // 2) Always check selling-unit prices after every batch addition
      await ensureSellUnitPrices(
        { shopId: currentItem.uid, categoryId: currentItem.categoryId, itemId: currentItem.itemId },
        priceResult.sellPrice ?? currentItem.data.sellPrice ?? 0,
        actor
      );

      // 3) Update local state and UI
      currentItem.data.stockTransactions = [
        ...(currentItem.data.stockTransactions || []),
        stockTransaction
      ];
      currentItem.data.batches = [
        ...batches,
        batch
      ];
      currentItem.data.stock = (currentItem.data.stock || 0) + quantity;

      if (priceResult.updateBuy) currentItem.data.buyPrice = priceResult.buyPrice;
      if (priceResult.updateSell) currentItem.data.sellPrice = priceResult.sellPrice;
      if (!currentItem.data.baseUnit) currentItem.data.baseUnit = baseUnit;

      let message = `✅ Added ${quantity} ${baseUnit}`;
      if (batches.length > 0) {
        message += `\n\nThis is batch #${batches.length + 1}`;
      }
      message += `\n\nTotal stock now: ${currentItem.data.stock} ${baseUnit}`;
      alert(message);

      renderItemMeta(currentItem.data);

    } catch (error) {
      console.error("❌ Error adding stock:", error);
      alert("Failed to add stock. Please try again.");
    }
  }

  // =========================================================
  // renderItemMeta
  // =========================================================
  function renderItemMeta(data = {}) {
    const imgs = data.images || [];
    const baseUnit = data.baseUnit || "units";
    const totalStock = data.stock || 0;
    const transactions = data.stockTransactions || [];
    const batches = data.batches || [];
    const lastThree = transactions.slice(-3).reverse();
    const canStock = canManageStock();

    if (itemNameEl) {
      if (editMode) {
        itemNameEl.innerHTML = `
          <input type="text" id="item-name-input" value="${data.name || ""}"
                 style="font-size: 1.5rem; padding: 5px; width: 80%; border: 1px solid #ccc; border-radius: 4px;">
          <span class="pencil" style="margin-left: 10px; color: #666;">✎</span>
        `;
      } else {
        itemNameEl.textContent = data.name || "";
      }
    }

    const imgHtml = [0, 1].map(i => {
      if (!imgs[i]) return `<div class="image-slot placeholder">No image</div>`;
      return `
        <div class="image-slot">
          <img src="${imgs[i]}" class="item-thumb">
          ${editMode ? `
            <div class="image-edit-overlay">
              <span class="pencil">✎</span>
              <button class="retake-btn" data-index="${i}">Retake</button>
            </div>
          ` : ""}
        </div>
      `;
    }).join("");

    let pricesHtml;
    if (editMode) {
      pricesHtml = `
        <div class="item-prices">
          <div style="margin-bottom: 8px; display: flex; align-items: center;">
            <span class="pencil" style="margin-right: 8px; color: #666;">✎</span>
            <label style="margin-right: 8px;">Buy Price (per ${baseUnit}):</label>
            <input type="number" id="buy-price-input" value="${data.buyPrice || ""}"
                   step="0.01" min="0" style="padding: 4px; width: 100px; border: 1px solid #ccc; border-radius: 4px;">
          </div>
          <div style="display: flex; align-items: center;">
            <span class="pencil" style="margin-right: 8px; color: #666;">✎</span>
            <label style="margin-right: 8px;">Sell Price (per ${baseUnit}):</label>
            <input type="number" id="sell-price-input" value="${data.sellPrice || ""}"
                   step="0.01" min="0" style="padding: 4px; width: 100px; border: 1px solid #ccc; border-radius: 4px;">
          </div>
        </div>
      `;
    } else {
      pricesHtml = `
        <div class="item-prices">
          <p>Buy Price: ${data.buyPrice ? `$${data.buyPrice} per ${baseUnit}` : "-"}</p>
          <p>Sell Price: ${data.sellPrice ? `$${data.sellPrice} per ${baseUnit}` : "-"}</p>
        </div>
      `;
    }

    const batchDisplay = batches.length > 0 ? `
      <div style="margin: 15px 0; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div style="font-size: 14px; font-weight: 600; color: #495057;">
            📦 Stock Batches (${batches.length})
          </div>
          <div style="font-size: 12px; color: #6c757d;">
            Each batch tracks price when added
          </div>
        </div>
        <div style="max-height: 100px; overflow-y: auto;">
          ${batches.slice(-2).reverse().map(batch => `
            <div style="padding: 8px; margin: 4px 0; background: white; border-radius: 6px; border-left: 3px solid #28a745;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; font-weight: 500;">${batch.date || ''}</span>
                <span style="font-weight: bold; color: #28a745;">+${batch.quantity} ${batch.unit}</span>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6c757d; margin-top: 2px;">
                <span>Buy: $${batch.buyPrice || 0}/${batch.unit}</span>
                <span>Sell: $${batch.sellPrice || 0}/${batch.unit}</span>
              </div>
            </div>
          `).join('')}
        </div>
        ${batches.length > 2 ? `
          <div style="text-align: center; margin-top: 8px;">
            <span style="font-size: 11px; color: #6c757d;">
              + ${batches.length - 2} earlier batches
            </span>
          </div>
        ` : ''}
      </div>
    ` : '';

    const stockHtml = `
      <div style="margin: 20px 0; padding: 15px; background: #f0f8ff; border: 2px solid #0077cc; border-radius: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
          <div>
            <div style="font-size: 16px; font-weight: bold; color: #0055aa;">📦 STOCK TRACKING</div>
            <div style="font-size: 32px; font-weight: bold; color: #0077cc;">
              ${totalStock}
              <span style="font-size: 16px; color: #666; margin-left: 5px;">${baseUnit}</span>
            </div>
            ${transactions.length > 0 ? `
              <div style="font-size: 12px; color: #888; margin-top: 5px;">
                Based on ${transactions.length} transaction${transactions.length === 1 ? "" : "s"}
                ${batches.length > 0 ? ` • ${batches.length} batch${batches.length === 1 ? "" : "es"}` : ''}
              </div>
            ` : ''}
          </div>

          <button id="add-stock-btn"
                  ${canStock ? "" : "disabled"}
                  style="
                    padding: 10px 20px;
                    background: ${canStock ? "linear-gradient(135deg, #0077cc, #0055aa)" : "#95a5a6"};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 16px;
                    cursor: ${canStock ? "pointer" : "not-allowed"};
                    font-weight: bold;
                    opacity: ${canStock ? "1" : "0.7"};
                  ">
            ➕ Add Stock
          </button>
        </div>

        ${batchDisplay}

        ${transactions.length > 0 ? `
          <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #cce7ff;">
            <div style="font-size: 14px; font-weight: bold; color: #0055aa; margin-bottom: 8px;">
              Recent Stock Movements:
            </div>
            <div style="max-height: 150px; overflow-y: auto;">
              ${lastThree.map(t => {
                const isSale = t.type === "sale" || Number(t.quantity) < 0;
                const label = isSale ? "Sold by" : "Added by";
                const who = t?.performedBy?.name || t?.performedBy?.email || t.addedBy || "Unknown";
                const qtyText = Number(t.quantity) > 0 ? `+${t.quantity}` : `${t.quantity}`;
                const qtyColor = isSale ? "#cc0000" : "#009900";
                const unit = t.unit || baseUnit;

                return `
                  <div style="padding: 10px; margin: 5px 0; background: white; border-radius: 6px;
                              border-left: 4px solid ${isSale ? "#cc0000" : "#0077cc"}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between;">
                      <span style="color: #333; font-weight: 500;">${t.date || ""}</span>
                      <span style="font-weight: bold; color: ${qtyColor}; font-size: 16px;">
                        ${qtyText} ${unit}
                      </span>
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">
                      ${label}: ${who}
                      ${t.buyPrice ? ` • Buy: $${t.buyPrice}/${unit}` : ''}
                      ${t.sellPrice ? ` • Sell: $${t.sellPrice}/${unit}` : ''}
                    </div>
                  </div>
                `;
              }).join("")}
            </div>

            ${transactions.length > 3 ? `
              <div style="text-align: center; margin-top: 10px;">
                <span style="font-size: 12px; color: #0077cc; font-style: italic;">
                  + ${transactions.length - 3} more records
                </span>
              </div>
            ` : ""}
          </div>
        ` : `
          <div style="margin-top: 15px; padding: 20px; text-align: center; color: #888; font-style: italic; background: white; border-radius: 6px;">
            No stock recorded yet.
          </div>
        `}
      </div>
    `;

    const alertThreshold = data.lowStockAlert || 5;
    const isAlertActive = totalStock <= alertThreshold;
    const bufferStatus = getBufferStatus(totalStock, alertThreshold);

    const alertHtml = `
      <div class="alert-slider-container ${isAlertActive ? "alert-active" : ""}">
        <div class="alert-slider-header">
          <div class="alert-title ${isAlertActive ? "alert-active" : ""}">
            ${isAlertActive ? "⚠️ LOW STOCK ALERT" : "📊 STOCK ALERTS"}
          </div>
          <div class="alert-status ${isAlertActive ? "alert-active" : ""}">
            ${isAlertActive ? "ALERT ACTIVE" : "MONITORING"}
          </div>
        </div>

        <div style="margin: 10px 0;">
          <div style="font-size: 14px; color: #666;">
            Alert when stock reaches: <span style="font-weight: bold; color: #333;">${alertThreshold}</span> ${baseUnit}
          </div>
          ${isAlertActive ? `
            <div style="font-size: 12px; color: #ff6b6b; margin-top: 4px; font-weight: bold;">
              ⚠️ Alert Active: Stock is at ${totalStock} ${baseUnit}!
            </div>
          ` : ""}
        </div>

        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.1);">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="stock-buffer ${bufferStatus.class}" style="font-size: 14px;">
              <span style="font-weight: 500;">Buffer:</span>
              <span style="font-weight: bold; margin-left: 5px;">
                ${totalStock - alertThreshold} ${baseUnit}
              </span>
              <span style="margin-left: 8px; font-size: 12px;">(${bufferStatus.text})</span>
            </div>

            ${editMode ? `
              <button id="configure-alert-btn" style="
                padding: 8px 16px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                cursor: pointer;
                font-weight: 500;
              ">
                Configure Alert
              </button>
            ` : `
              <div style="padding: 8px 12px; background: #f8f9fa; border-radius: 6px; color: #666; font-size: 12px;">
                ${canStock ? "Edit mode to change" : "No permission to change"}
              </div>
            `}
          </div>
        </div>

        ${editMode ? `
          <div id="alert-config-panel" class="alert-config-panel">
            <div style="margin-bottom: 15px;">
              <div style="font-weight: 600; color: #333; margin-bottom: 8px;">
                Set Low Stock Alert Level
              </div>

              <input type="range"
                     id="alert-slider"
                     min="0"
                     max="50"
                     value="${alertThreshold}"
                     step="1"
                     class="alert-slider">

              <div id="alert-value-display" class="alert-value-display">${alertThreshold}</div>

              <div class="alert-labels">
                <span>0 (No Alert)</span>
                <span>Critical: 5</span>
                <span>Warning: 10</span>
                <span>Safe: 25</span>
                <span>50+</span>
              </div>
            </div>

            <div style="margin-bottom: 15px;">
              <label style="display: block; margin-bottom: 8px; color: #666; font-size: 14px;">
                Alert Description (Optional)
              </label>
              <input type="text"
                     id="alert-description"
                     placeholder="e.g., 'Restock immediately'"
                     value="${data.alertDescription || ""}"
                     class="alert-description-input">
            </div>

            <div class="alert-buttons">
              <button id="save-alert-btn" class="alert-save-btn">Save Alert</button>
              <button id="cancel-alert-btn" class="alert-cancel-btn">Cancel</button>
            </div>
          </div>
        ` : ""}
      </div>
    `;

    itemMeta.innerHTML = `
      <div class="item-images">${imgHtml}</div>
      ${pricesHtml}
      ${stockHtml}
      ${alertHtml}
      <div class="capture-actions"></div>
      <div class="capture-status"></div>
    `;

    const addStockBtn = itemMeta.querySelector("#add-stock-btn");
    if (addStockBtn) addStockBtn.addEventListener("click", addStockToItem);

    if (editMode) {
      setTimeout(() => updateAlertSliderValue("alert-slider", "alert-value-display"), 50);

      const configureBtn = itemMeta.querySelector("#configure-alert-btn");
      const saveAlertBtn = itemMeta.querySelector("#save-alert-btn");
      const cancelAlertBtn = itemMeta.querySelector("#cancel-alert-btn");
      const alertConfigPanel = itemMeta.querySelector("#alert-config-panel");
      const alertSlider = itemMeta.querySelector("#alert-slider");

      if (configureBtn && alertConfigPanel) {
        configureBtn.addEventListener("click", () => {
          alertConfigPanel.style.display =
            alertConfigPanel.style.display === "none" || !alertConfigPanel.style.display
              ? "block" : "none";
        });
      }

      if (saveAlertBtn) saveAlertBtn.addEventListener("click", saveAlertSettings);

      if (cancelAlertBtn && alertConfigPanel) {
        cancelAlertBtn.addEventListener("click", () => {
          alertConfigPanel.style.display = "none";
        });
      }

      if (alertSlider) {
        alertSlider.addEventListener("input", () => {
          updateAlertSliderValue("alert-slider", "alert-value-display");
        });
      }

      bindRetakeButtons();
    }
  }

  // =========================================================
  // RETAKE IMAGE
  // =========================================================
  function bindRetakeButtons() {
    if (!canManageStock()) return;

    itemMeta.querySelectorAll(".retake-btn").forEach(btn => {
      btn.onclick = async () => {
        if (!canManageStock()) return;

        const actor = getActor();
        const index = Number(btn.dataset.index);
        showStatus(`Retaking image ${index + 1}…`);

        const file = await promptCameraCapture();
        if (!file) return clearStatus();

        const preview = URL.createObjectURL(file);
        setPreviewImageSlot(index, preview);

        const url = await uploadToCloudinary(file);
        currentItem.data.images[index] = url;

        const itemRef = doc(
          db,
          "Shops", currentItem.uid,
          "categories", currentItem.categoryId,
          "items", currentItem.itemId
        );

        await updateDoc(itemRef, {
          images: currentItem.data.images,
          updatedAt: Date.now(),
          updatedBy: actor
        });

        sendImageForEmbedding(url, index);
        clearStatus();
        renderItemMeta(currentItem.data);
      };
    });
  }

  // =========================================================
  // HELPERS
  // =========================================================
  function promptCameraCapture() {
    return new Promise(resolve => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = "image/*";
      input.capture = "environment";
      input.style.display = "none";

      captureInProgress = true;

      input.onchange = () => {
        captureInProgress = false;
        resolve(input.files?.[0] || null);
        input.remove();
      };

      document.body.appendChild(input);
      input.click();
    });
  }

  function setPreviewImageSlot(index, url) {
    const slot = itemMeta.querySelector(`.image-slot:nth-child(${index + 1})`);
    if (slot) slot.innerHTML = `<img src="${url}" class="item-thumb">`;
  }

  function setPlaceholderMessage(index, text) {
    const slot = itemMeta.querySelector(`.image-slot:nth-child(${index + 1})`);
    if (slot) slot.textContent = text;
  }

  async function uploadToCloudinary(file) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("upload_preset", CLOUDINARY_UPLOAD_PRESET);

    const res = await fetch(
      `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD}/image/upload`,
      { method: "POST", body: fd }
    );

    return (await res.json()).secure_url;
  }

  function showStatus(text) {
    const s = itemMeta.querySelector(".capture-status");
    if (s) s.textContent = text;
  }

  function clearStatus() {
    const s = itemMeta.querySelector(".capture-status");
    if (s) s.textContent = "";
  }

  // =========================================================
  // HIDE ITEM DETAIL
  // =========================================================
  function hideItemDetail() {
    itemDetail.classList.add("hidden");
    overlayContent.classList.remove("hidden");
    document.body.style.overflow = "";
    currentItem = null;
    editMode = false;

    if (editToggleBtn) editToggleBtn.textContent = "Edit";
    syncEditButtonUI();

    const closeBtn = document.getElementById("item-detail-close-btn");
    if (closeBtn) closeBtn.remove();

    if (itemDetail) {
      ["shopId","categoryId","itemId","baseUnit","itemName"].forEach(k => delete itemDetail.dataset[k]);
    }
    document.dispatchEvent(new CustomEvent("item-detail:closed"));
  }

  // =========================================================
  // PUBLIC API
  // =========================================================
  window.attachItemDetailHandler = (el, name, uid, categoryId, itemId) => {
    el.onclick = e => {
      e.stopPropagation();
      showItemDetail(name, uid, categoryId, itemId);
    };
  };

  window.getCurrentItemContext = () => {
    if (!currentItem) return null;
    return {
      shopId: currentItem.uid,
      categoryId: currentItem.categoryId,
      itemId: currentItem.itemId,
      itemName: currentItem.data?.name || currentItem.name || "",
      baseUnit: currentItem.data?.baseUnit || itemDetail?.dataset?.baseUnit || "unit"
    };
  };

  window.hideItemDetail = hideItemDetail;
});