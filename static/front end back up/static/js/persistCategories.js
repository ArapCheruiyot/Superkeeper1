// persistCategories.js

// --- Firebase imports using browser-safe CDN modules ---
import { db } from "./firebase-config.js";

import {
    collection,
    getDocs,
    query,
    where
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

/**
 * Fetch all items for a given shop from Firestore
 * @param {string} shopId
 * @returns {Promise<Array>} items
 */
export async function fetchShopItems(shopId) {
    const itemsCol = collection(db, "Items");
    const q = query(itemsCol, where("shopId", "==", shopId));
    const snapshot = await getDocs(q);

    const items = [];
    snapshot.forEach(doc => items.push(doc.data()));
    return items;
}

/**
 * Build a nested tree structure for categories & subcategories from items
 * @param {Array} items
 * @returns {Object} category tree
 */
export function buildCategoryTree(items) {
    const tree = {};

    items.forEach(item => {
        const path = item.categoryPath.split("/"); // support nested paths
        let current = tree;

        path.forEach((level, idx) => {
            if (!current[level]) current[level] = { items: [], subcategories: {} };

            // Last level → add item
            if (idx === path.length - 1) {
                current[level].items.push(item.itemName);
            }

            current = current[level].subcategories;
        });
    });

    return tree;
}

/**
 * Recursively render the category tree into the DOM
 * @param {Object} tree
 * @param {HTMLElement} parentContainer
 * @param {Function} createCategoryElement - your existing function
 * @param {number} level
 */
export function renderCategoryTree(tree, parentContainer, createCategoryElement, level = 0) {
    Object.entries(tree).forEach(([name, data]) => {
        const catEl = createCategoryElement(name, level);
        parentContainer.appendChild(catEl);

        // Add items inside this category
        data.items.forEach(itemName => {
            const item = document.createElement("div");
            item.textContent = itemName;
            item.className = "font-bold p-1 category-item";
            catEl._childrenContainer.appendChild(item);
        });

        // Render subcategories recursively
        renderCategoryTree(data.subcategories, catEl._childrenContainer, createCategoryElement, level + 1);
    });
}

/**
 * Load categories from Firestore and render them
 * @param {string} shopId
 * @param {HTMLElement} listContainer
 * @param {Function} createCategoryElement
 */
export async function loadPersistedCategories(shopId, listContainer, createCategoryElement) {
    try {
        const items = await fetchShopItems(shopId);
        const tree = buildCategoryTree(items);
        renderCategoryTree(tree, listContainer, createCategoryElement);
    } catch (error) {
        console.error("Error loading categories:", error);
    }
}
