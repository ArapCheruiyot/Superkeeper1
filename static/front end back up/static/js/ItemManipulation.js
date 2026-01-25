// ItemManipulation.js — FULL PAGE OVERLAY + Item Metadata + Modular Editing

import { attachItemNameEditor } from "./itemDetails/itemNameEditor.js";
import { ensureItemImages } from "./itemDetails/itemImageCapture.js";
import { ensureItemMetadata } from "./itemDetails/itemAdditionalMetadata.js";
import { attachItemSyncManager } from "./itemDetails/ItemSyncManager.js"; // <-- NEW MODULE

document.addEventListener("DOMContentLoaded", () => {
    console.log("ItemManipulation ACTIVE ✔");
    injectItemStyles();
    setupItemClickListener();
});

/* -------------------------------------------------------
   ADD CLICK LISTENER TO REAL ITEMS ONLY
------------------------------------------------------- */
function setupItemClickListener() {
    document.body.addEventListener("click", function (event) {
        const clickedItem = event.target.closest(".category-item");

        if (
            !clickedItem ||
            !clickedItem.parentElement.classList.contains("children-container")
        ) return;

        const itemName = clickedItem.textContent.trim();
        console.log("Item clicked:", itemName);

        openItemOverlay(itemName);
    });
}

/* -------------------------------------------------------
   INJECT OVERLAY + ITEM STYLES
------------------------------------------------------- */
function injectItemStyles() {
    const style = document.createElement("style");
    style.textContent = `
        .category-item {
            cursor: pointer;
            transition: 0.25s ease-in-out;
            padding: 4px;
            border-radius: 4px;
        }
        .category-item:hover {
            background-color: #dbeafe;
            padding-left: 10px;
        }

        #item-overlay {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(0, 0, 0, 0.6) !important;
            display: none;
            justify-content: center;
            align-items: flex-start;
            padding-top: 80px;
            z-index: 999999 !important;
            backdrop-filter: blur(3px);
            overflow-y: auto;
        }

        #item-overlay-content {
            background: white;
            padding: 24px;
            width: 100%;
            height: calc(100vh - 80px);
            box-sizing: border-box;
            overflow-y: auto;
            transition: 0.25s ease-out;
            position: relative;
        }

        #close-item-overlay {
            position: fixed;
            top: 20px;
            right: 20px;
            cursor: pointer;
            font-size: 2rem;
            background: transparent;
            border: none;
            z-index: 1000000;
            color: #ef4444;
        }

        .overlay-image-container img {
            width: 100%;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        .meta-block {
            margin-top: 20px;
            padding: 12px;
            background: #f3f4f6;
            border-radius: 6px;
        }

        .sync-button {
            position: absolute;
            top: 16px;
            right: 16px;
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background-color: #3b82f6;
            color: white;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);
}

/* -------------------------------------------------------
   OPEN ITEM OVERLAY
------------------------------------------------------- */
function openItemOverlay(itemName) {
    let overlay = document.getElementById("item-overlay");

    // Create overlay only once
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "item-overlay";

        overlay.innerHTML = `
            <div id="item-overlay-content">

                <div class="mb-4" style="position: relative;">
                    <h2 id="overlay-item-name" class="text-2xl font-bold text-blue-900"></h2>
                    <!-- Sync/Edit button injected here -->
                </div>

                <!-- Image Area -->
                <div id="item-image-section" class="text-gray-700 text-lg"></div>

                <!-- Metadata Area -->
                <div id="item-metadata-section" class="meta-block"></div>

            </div>

            <button id="close-item-overlay">&times;</button>
        `;

        document.body.appendChild(overlay);

        // Close overlay
        overlay.querySelector("#close-item-overlay").addEventListener("click", () => {
            overlay.style.display = "none";
            overlay.classList.remove("show");
        });
    }

    // Set item name
    document.getElementById("overlay-item-name").textContent = itemName;

    // Show overlay
    overlay.style.display = "flex";
    requestAnimationFrame(() => overlay.classList.add("show"));

    // Name editor
    attachItemNameEditor();

    // Ensure images exist
    ensureItemImages(itemName);

    // Ensure metadata (buying & selling price)
    ensureItemMetadata(itemName, document.getElementById("item-metadata-section"));

    // Attach Item Sync Manager (Edit/Save Changes button)
    attachItemSyncManager(itemName);
}

export { openItemOverlay };
