// ItemSyncManager.js
// Prepares Firestore-safe payload and renders items from Firestore

import { auth, db } from "../firebase-config.js"; // Ensure 'db' is exported from firebase-config

export function attachItemSyncManager(itemName) {
    const overlay = document.getElementById("item-overlay-content");
    if (!overlay) return;

    // -------------------- Helpers --------------------
    function normalizeForLookup(name) {
        return name.replace(/[^a-zA-Z0-9 ]/g, "").trim().toLowerCase();
    }

    function safeName(name) {
        return name.replace(/[^\w\s]/g, "").trim();
    }

    const ORIGINAL_CLEAN_NAME = normalizeForLookup(itemName);

    function computeCategoryPathByOriginalName() {
        const allItems = Array.from(document.querySelectorAll(".category-item"));
        const found = allItems.find(el => normalizeForLookup(el.textContent) === ORIGINAL_CLEAN_NAME);
        if (!found) return [];

        let ancestor = found.closest(".category-block[data-level]");
        const path = [];
        while (ancestor) {
            const titleEl = ancestor.querySelector(".category-item");
            if (titleEl && titleEl.textContent) path.unshift(titleEl.textContent.trim());
            ancestor = ancestor.parentElement?.closest(".category-block[data-level]") || null;
        }
        return path;
    }

    // -------------------- Wait for overlay DOM --------------------
    function waitForRender() {
        const priceRows = overlay.querySelectorAll(".price-row");
        if (priceRows.length === 0) return setTimeout(waitForRender, 40);
        setupManager(Array.from(priceRows));
    }

    // -------------------- Setup Manager --------------------
    function setupManager(priceRows) {
        // Header & Edit Button
        let header = overlay.querySelector("#overlay-item-name-container");
        if (!header) {
            const nameElem = document.getElementById("overlay-item-name");
            header = document.createElement("div");
            header.id = "overlay-item-name-container";
            header.style.display = "flex";
            header.style.justifyContent = "space-between";
            header.style.alignItems = "center";
            header.style.marginBottom = "12px";
            nameElem.parentNode.insertBefore(header, nameElem);
            header.appendChild(nameElem);
        }

        let btn = header.querySelector("#edit-save-btn");
        if (!btn) {
            btn = document.createElement("button");
            btn.id = "edit-save-btn";
            btn.textContent = "Edit";
            btn.style.padding = "6px 14px";
            btn.style.border = "none";
            btn.style.borderRadius = "5px";
            btn.style.background = "#3b82f6";
            btn.style.color = "white";
            btn.style.cursor = "pointer";
            header.appendChild(btn);
        }

        // Lock/unlock overlay controls
        function lock() {
            overlay.querySelectorAll(".price-edit-btn, .capture-camera-btn").forEach(b => b.disabled = true);
            const nameEditIcon = overlay.querySelector(".edit-icon");
            if (nameEditIcon) nameEditIcon.style.pointerEvents = "none";
        }

        function unlock() {
            overlay.querySelectorAll(".price-edit-btn, .capture-camera-btn").forEach(b => b.disabled = false);
            const nameEditIcon = overlay.querySelector(".edit-icon");
            if (nameEditIcon) nameEditIcon.style.pointerEvents = "auto";
        }

        // Inline editing for prices
        function enableInlineEditing(row) {
            const valueSpan = row.querySelector(".price-value");
            if (!valueSpan) return;
            const oldValue = valueSpan.textContent.replace("Not set", "").trim();
            const input = document.createElement("input");
            input.type = "number";
            input.value = oldValue || "";
            input.className = "price-edit-input";
            input.style.padding = "6px";
            input.style.width = "100px";
            input.style.border = "1px solid #aaa";
            input.style.borderRadius = "4px";
            valueSpan.replaceWith(input);
            input.focus();
            input.select();
        }

        function closeEditing(row) {
            const input = row.querySelector(".price-edit-input");
            if (!input) return;
            const span = document.createElement("span");
            span.className = "price-value";
            span.style.fontWeight = "bold";
            span.textContent = input.value ? input.value : "Not set";
            input.replaceWith(span);
        }

        // Setup edit buttons for each price row
        priceRows.forEach(row => {
            const editBtn = row.querySelector(".price-edit-btn");
            if (!editBtn) return;
            editBtn.onclick = () => {
                if (btn.textContent !== "Save Changes") {
                    alert("Click Edit first to make changes.");
                    return;
                }
                enableInlineEditing(row);
            };
        });

        // Validate before sending
        function validate() {
            const inputs = overlay.querySelectorAll(".price-edit-input");
            if (inputs.length > 0) {
                for (let input of inputs) {
                    if (!input.value.trim()) {
                        alert("Both buying and selling prices must be entered.");
                        return false;
                    }
                }
            } else {
                const values = overlay.querySelectorAll(".price-value");
                for (let v of values) {
                    if (!v.textContent.trim() || v.textContent.trim() === "Not set") {
                        alert("Both buying and selling prices must be entered.");
                        return false;
                    }
                }
            }

            const images = overlay.querySelectorAll(".overlay-image-container img");
            if (images.length < 2) {
                alert("Please capture at least 2 images.");
                return false;
            }
            return true;
        }

        // Collect Firestore-safe payload
        function collect() {
            const rows = overlay.querySelectorAll(".price-row");

            function valueFrom(row) {
                const input = row.querySelector(".price-edit-input");
                if (input) return Number(input.value) || 0;
                const label = row.querySelector(".price-value");
                const val = label ? label.textContent.trim() : "0";
                return Number(val) || 0;
            }

            const rawTitle = document.getElementById("overlay-item-name").textContent.trim();
            const cleanedTitle = safeName(rawTitle);
            const categoryPathArray = computeCategoryPathByOriginalName();
            const categoryPathString = Array.isArray(categoryPathArray) ? categoryPathArray.join(" > ") : "";
            const shopId = auth.currentUser ? auth.currentUser.uid : null;

            const imageEls = overlay.querySelectorAll(".overlay-image-container img");
            const imageUrls = Array.from(imageEls).map(i => i.src);

            return {
                shopId,
                itemName: cleanedTitle,
                categoryPath: categoryPathString,
                buyingPrice: valueFrom(rows[0]),
                sellingPrice: valueFrom(rows[1]),
                images: imageUrls,
                textVector: [],
                imageVectors: {},
                updatedAt: new Date().toISOString()
            };
        }

        lock();

        // -------------------- Button Click --------------------
        btn.onclick = async () => {
            if (btn.textContent === "Edit") {
                unlock();
                btn.textContent = "Save Changes";
                btn.style.background = "#16a34a";
            } else {
                if (!validate()) return;
                priceRows.forEach(r => closeEditing(r));
                const payload = collect();
                console.log("🔥 Payload ready for backend:", payload);

                try {
                    const response = await fetch("/itemEmbeder", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();
                    console.log("✅ Backend response:", data);
                } catch (err) {
                    console.error("❌ Error sending payload:", err);
                }

                lock();
                btn.textContent = "Edit";
                btn.style.background = "#3b82f6";
            }
        };

        // -------------------- Load Items from Firestore --------------------
        function loadItems() {
            const shopId = auth.currentUser ? auth.currentUser.uid : null;
            if (!shopId) return;
            const container = document.getElementById("items-container");
            if (!container) return;

            const itemsRef = db.collection("Shops").doc(shopId).collection("Items");
            itemsRef.onSnapshot(snapshot => {
                container.innerHTML = ""; // clear old items
                snapshot.forEach(doc => {
                    const data = doc.data();
                    const itemEl = document.createElement("div");
                    itemEl.className = "item-card";
                    itemEl.innerHTML = `
                        <h3>${data.itemName}</h3>
                        <p>Category: ${data.categoryPath}</p>
                        <p>Buy: ${data.buyingPrice}, Sell: ${data.sellingPrice}</p>
                        ${data.images[0] ? `<img src="${data.images[0]}" width="100" />` : ""}
                    `;
                    container.appendChild(itemEl);
                });
            });
        }

        loadItems(); // call once to persist items
    }

    waitForRender();
}
