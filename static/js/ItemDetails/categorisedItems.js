// categorisedItems.js
import { db } from "./firebase-config.js";
import {
  collection,
  getDocs,
  getDoc,
  addDoc,
  updateDoc,
  deleteDoc,
  doc,
  query,
  where
} from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

document.addEventListener("DOMContentLoaded", () => {
  // UI references
  const manageStockBtn = document.getElementById("manage-stock-btn");
  const overlay = document.getElementById("overlay");
  const overlayContent = document.querySelector(".overlay-content");
  const categoriesBtn = document.getElementById("categories-btn");
  const categoriesList = document.getElementById("categories-list");

  const categoryModal = document.getElementById("category-modal");
  const modalTitle = document.getElementById("modal-title");
  const addSubBtn = document.getElementById("add-subcategory-btn");
  const addItemBtn = document.getElementById("add-item-btn");
  const deleteCatBtn = document.getElementById("delete-category-btn");
  const closeModalX = document.getElementById("close-modal-x");

  // Item detail overlay (exists in HTML)
  const itemDetail = document.getElementById("item-detail");

  // State
  let currentCategory = null;
  let currentNodeData = null;
  let currentUserId = null;

  /* ------------------------------
     Firestore path helpers
  -------------------------------*/
  function categoriesCollectionPath(uid) {
    return ["Shops", uid, "categories"];
  }
  function itemsCollectionPath(uid, categoryId) {
    return ["Shops", uid, "categories", categoryId, "items"];
  }

  /* ------------------------------
     Attach item handler (robust)
  -------------------------------*/
  function attachItemHandlerWithRetry(el, name, uid, categoryId, itemId) {
    const MAX_ATTEMPTS = 12;
    let attempts = 0;

    function fallback() {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        alert(`Item clicked: ${name}`);
      });
      console.warn("attachItemDetailHandler not found after retries. Falling back to alert for item:", name);
    }

    function tryAttach() {
      attempts++;
      if (window && typeof window.attachItemDetailHandler === "function") {
        try {
          window.attachItemDetailHandler(el, name, uid, categoryId, itemId);
        } catch (err) {
          console.error("attachItemDetailHandler threw:", err);
          fallback();
        }
      } else if (attempts > MAX_ATTEMPTS) {
        fallback();
      } else {
        setTimeout(tryAttach, 150);
      }
    }

    tryAttach();
  }

  /* ------------------------------
     Create category DOM node
  -------------------------------*/
  function createCategoryNode(name, id) {
    const el = document.createElement("div");
    el.className = "category-item";
    el.textContent = name;
    el.dataset.id = id;

    const children = document.createElement("div");
    children.className = "children";
    el.appendChild(children);

    el.addEventListener("click", (e) => {
      e.stopPropagation();
      currentCategory = el;
      currentNodeData = { id, name };
      if (modalTitle) modalTitle.textContent = `Category: ${name}`;
      showModal();
    });

    return el;
  }

  /* ------------------------------
     Modal helpers
  -------------------------------*/
  function showModal() {
    if (!categoryModal) return;
    categoryModal.classList.remove("hidden");
    updateModalButtons();
  }

  function hideModal() {
    if (!categoryModal) return;
    categoryModal.classList.add("hidden");
    currentCategory = null;
    currentNodeData = null;
  }

  closeModalX?.addEventListener("click", hideModal);

  function updateModalButtons() {
    if (!currentCategory) return;
    const children = currentCategory.querySelector(".children").children;
    const hasSubcategories = Array.from(children).some(c => c.classList.contains("category-item"));
    const hasItems = Array.from(children).some(c => c.classList.contains("item"));
    addSubBtn.style.display = hasItems ? "none" : "inline-block";
    addItemBtn.style.display = hasSubcategories ? "none" : "inline-block";
  }

  /* ------------------------------
     Overlay helpers (stack behavior)
  -------------------------------*/
  function showCategoriesOverlay() {
    overlay.classList.remove("hidden");
    overlayContent.classList.remove("hidden");
    // ensure item detail is hidden when showing categories
    if (itemDetail) {
      itemDetail.classList.add("hidden");
      itemDetail.setAttribute("aria-hidden", "true");
    }
  }

  function showItemDetailOverlay() {
    if (itemDetail) {
      itemDetail.classList.remove("hidden");
      itemDetail.setAttribute("aria-hidden", "false");
    }
  }

  function closeItemDetailOverlay() {
    if (itemDetail) {
      itemDetail.classList.add("hidden");
      itemDetail.setAttribute("aria-hidden", "true");
    }
    // categories overlay remains visible
  }

  function hideOverlayCompletely() {
    overlay.classList.add("hidden");
    overlayContent.classList.add("hidden");
    if (itemDetail) {
      itemDetail.classList.add("hidden");
      itemDetail.setAttribute("aria-hidden", "true");
    }
    hideModal();
  }

  /* ------------------------------
     Inject dynamic close buttons
     (so we don't depend on static HTML)
  -------------------------------*/
  function injectOverlayCloseButtons() {
    // categories overlay close
    if (overlay && !document.getElementById("dynamic-close-overlay")) {
      const closeBtn = document.createElement("span");
      closeBtn.id = "dynamic-close-overlay";
      closeBtn.className = "close-x";
      closeBtn.setAttribute("role", "button");
      closeBtn.setAttribute("aria-label", "Close categories overlay");
      closeBtn.innerHTML = "&times;";
      closeBtn.style.position = "absolute";
      closeBtn.style.top = "10px";
      closeBtn.style.right = "15px";
      closeBtn.style.fontSize = "24px";
      closeBtn.style.cursor = "pointer";
      overlay.appendChild(closeBtn);

      closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        hideOverlayCompletely();
      });
    }

    // item detail overlay close
    if (itemDetail && !document.getElementById("dynamic-close-item-detail")) {
      const closeBtn = document.createElement("span");
      closeBtn.id = "dynamic-close-item-detail";
      closeBtn.className = "close-x";
      closeBtn.setAttribute("role", "button");
      closeBtn.setAttribute("aria-label", "Close item detail overlay");
      closeBtn.innerHTML = "&times;";
      closeBtn.style.position = "absolute";
      closeBtn.style.top = "10px";
      closeBtn.style.right = "15px";
      closeBtn.style.fontSize = "24px";
      closeBtn.style.cursor = "pointer";
      itemDetail.appendChild(closeBtn);

      closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        closeItemDetailOverlay();
      });
    }
  }

  /* ------------------------------
     Load categories and nested items
     (defined before auth watcher)
  -------------------------------*/
  async function loadCategories() {
    if (!currentUserId) return;
    categoriesList.innerHTML = "";

    const catSnap = await getDocs(collection(db, ...categoriesCollectionPath(currentUserId)));
    const map = {};

    catSnap.forEach(d => {
      const data = d.data();
      map[d.id] = {
        node: createCategoryNode(data.name, d.id),
        parentId: data.parentId
      };
    });

    // Build tree
    Object.values(map).forEach(({ node, parentId }) => {
      if (parentId && map[parentId]) {
        map[parentId].node.querySelector(".children").appendChild(node);
      } else {
        categoriesList.appendChild(node);
      }
    });

    // Load items for each category
    for (const catId of Object.keys(map)) {
      const itemsSnap = await getDocs(collection(db, ...itemsCollectionPath(currentUserId, catId)));
      itemsSnap.forEach(d => {
        const data = d.data();
        const parent = map[catId]?.node;
        if (!parent) return;

        const item = document.createElement("div");
        item.className = "item";
        item.textContent = data.name;
        item.dataset.id = d.id;

        // attach overlay handler robustly
        attachItemHandlerWithRetry(item, data.name, currentUserId, catId, d.id);

        parent.querySelector(".children").appendChild(item);
      });
    }

    // Ensure dynamic close buttons exist after categories are rendered
    injectOverlayCloseButtons();
  }

  // Expose reload function for other modules
  window.reloadShopCategories = loadCategories;

  /* ------------------------------
     Auth watcher (calls loadCategories)
  -------------------------------*/
  const auth = getAuth();
  auth.onAuthStateChanged(user => {
    if (user) {
      currentUserId = user.uid;
      loadCategories().catch(err => console.error("Failed to load categories:", err));
    } else {
      currentUserId = null;
      if (categoriesList) categoriesList.innerHTML = "";
      console.warn("User not signed in. Categories will not load until sign-in.");
    }
  });

  /* ------------------------------
     Event bindings
  -------------------------------*/
  // Manage Stock always shows categories overlay
  manageStockBtn?.addEventListener("click", () => {
    showCategoriesOverlay();
  });

  // Backdrop click closes everything
  overlay?.addEventListener("click", (e) => {
    if (e.target === overlay) {
      hideOverlayCompletely();
    }
  });

  /* ------------------------------
     Remaining CRUD helpers (saveCategory, saveItem, nameExists, updateName, rebuildAllCategoryPaths, etc.)
     These are unchanged from your original file but included here for completeness.
  -------------------------------*/
  async function saveCategory(name, parentId = null) {
    if (!currentUserId) return null;
    let ancestors = [];
    let fullPath = name;

    if (parentId) {
      const parentRef = doc(db, "Shops", currentUserId, "categories", parentId);
      const parentSnap = await getDoc(parentRef);
      if (!parentSnap.exists()) {
        throw new Error("Parent category not found");
      }
      const parent = parentSnap.data();
      ancestors = Array.isArray(parent.ancestors) ? [...parent.ancestors] : [];
      ancestors.push({ id: parentId, name: parent.name });
      fullPath = ancestors.map(a => a.name).concat(name).join(" > ");
    }

    const ref = await addDoc(collection(db, ...categoriesCollectionPath(currentUserId)), {
      name,
      parentId,
      ancestors,
      fullPath,
      createdAt: Date.now()
    });

    return ref.id;
  }

  async function saveItem(name, parentId, itemData = {}) {
    if (!currentUserId) return null;
    const catRef = doc(db, "Shops", currentUserId, "categories", parentId);
    const catSnap = await getDoc(catRef);
    if (!catSnap.exists()) throw new Error("Category not found");

    const cat = catSnap.data();
    const ancestors = Array.isArray(cat.ancestors) ? [...cat.ancestors] : [];
    ancestors.push({ id: parentId, name: cat.name });
    const fullPath = ancestors.map(a => a.name).concat(name).join(" > ");

    const ref = await addDoc(collection(db, ...itemsCollectionPath(currentUserId, parentId)), {
      name,
      categoryId: parentId,
      ancestors,
      fullPath,
      ...itemData,
      createdAt: Date.now()
    });

    return ref.id;
  }

  async function nameExistsInCollection(collectionPath, name) {
    if (!currentUserId) return { exists: false };
    const colRef = collection(db, ...collectionPath);
    const snap = await getDocs(colRef);
    const key = name.trim().toLowerCase();
    const existingDoc = snap.docs.find(d => (d.data().name || "").toLowerCase() === key);
    return existingDoc ? { exists: true, docId: existingDoc.id, data: existingDoc.data() } : { exists: false };
  }

  async function updateNameInCollection(collectionPath, docId, newName) {
    if (!currentUserId) return;
    await updateDoc(doc(db, ...collectionPath, docId), { name: newName, updatedAt: Date.now() });
    if (collectionPath.length >= 3 && collectionPath[collectionPath.length - 1] === "categories") {
      await rebuildAllCategoryPaths(currentUserId);
    }
  }

  async function rebuildAllCategoryPaths(uid) {
    const catSnap = await getDocs(collection(db, ...categoriesCollectionPath(uid)));
    const map = {};
    catSnap.forEach(d => map[d.id] = { id: d.id, ...d.data() });

    function computeAncestorsAndPath(catId) {
      const ancestors = [];
      let cur = map[catId];
      while (cur && cur.parentId) {
        const parent = map[cur.parentId];
        if (!parent) break;
        ancestors.unshift({ id: parent.id, name: parent.name });
        cur = parent;
      }
      const fullPath = ancestors.map(a => a.name).concat(map[catId].name).join(" > ");
      return { ancestors, fullPath };
    }

    for (const id of Object.keys(map)) {
      const { ancestors, fullPath } = computeAncestorsAndPath(id);
      await updateDoc(doc(db, "Shops", uid, "categories", id), { ancestors, fullPath, updatedAt: Date.now() });
    }

    for (const id of Object.keys(map)) {
      const cat = map[id];
      const catAncestors = Array.isArray(cat.ancestors) ? [...cat.ancestors] : [];
      const itemsSnap = await getDocs(collection(db, ...itemsCollectionPath(uid, id)));
      for (const itemDoc of itemsSnap.docs) {
        const item = itemDoc.data();
        const itemAncestors = [...catAncestors, { id, name: cat.name }];
        const itemFullPath = itemAncestors.map(a => a.name).concat(item.name).join(" > ");
        await updateDoc(doc(db, "Shops", uid, "categories", id, "items", itemDoc.id), {
          ancestors: itemAncestors,
          fullPath: itemFullPath,
          updatedAt: Date.now()
        });
      }
    }
  }

  /* ------------------------------
     Category/item creation and deletion bindings (unchanged)
     These use the helpers above and will continue to work.
  -------------------------------*/
  categoriesBtn?.addEventListener("click", async () => {
    if (!currentUserId) {
      alert("Please sign in to manage categories.");
      return;
    }

    const name = prompt("Enter category name:");
    if (!name?.trim()) return;
    const clean = name.trim();

    const { exists, docId } = await nameExistsInCollection(categoriesCollectionPath(currentUserId), clean);
    if (exists) {
      const confirmEdit = confirm(`Category "${clean}" already exists. Do you want to rename it?`);
      if (!confirmEdit) return;
      const newName = prompt("Enter new name for category:", clean);
      if (!newName?.trim()) return;
      await updateNameInCollection(categoriesCollectionPath(currentUserId), docId, newName.trim());
      await loadCategories();
      return;
    }

    try {
      const id = await saveCategory(clean);
      if (!id) return;
      const node = createCategoryNode(clean, id);
      categoriesList.appendChild(node);
    } catch (err) {
      console.error("Failed to create category", err);
      alert("Failed to create category. See console for details.");
    }
  });

  addSubBtn?.addEventListener("click", async () => {
    if (!currentCategory || !currentUserId) return;

    const name = prompt("Enter subcategory name:");
    if (!name?.trim()) return;
    const clean = name.trim();

    const { exists, docId } = await nameExistsInCollection(categoriesCollectionPath(currentUserId), clean);
    if (exists) {
      const confirmEdit = confirm(`Subcategory "${clean}" already exists. Do you want to rename it?`);
      if (!confirmEdit) return;
      const newName = prompt("Enter new name for subcategory:", clean);
      if (!newName?.trim()) return;
      await updateNameInCollection(categoriesCollectionPath(currentUserId), docId, newName.trim());
      await loadCategories();
      return;
    }

    try {
      const id = await saveCategory(clean, currentCategory.dataset.id);
      if (!id) return;
      const node = createCategoryNode(clean, id);
      currentCategory.querySelector(".children").appendChild(node);
      hideModal();
    } catch (err) {
      console.error("Failed to create subcategory", err);
      alert("Failed to create subcategory. See console for details.");
    }
  });

  addItemBtn?.addEventListener("click", async () => {
    if (!currentCategory || !currentUserId) return;

    const isLeaf = await isLeafCategory(currentUserId, currentCategory.dataset.id);
    if (!isLeaf) {
      alert("This category has subcategories. Add items only to a leaf category.");
      return;
    }

    const name = prompt("Enter item name:");
    if (!name?.trim()) return;
    const clean = name.trim();

    const itemsPath = itemsCollectionPath(currentUserId, currentCategory.dataset.id);
    const { exists, docId } = await nameExistsInCollection(itemsPath, clean);
    if (exists) {
      const confirmEdit = confirm(`Item "${clean}" already exists. Do you want to rename it?`);
      if (!confirmEdit) return;
      const newName = prompt("Enter new name for item:", clean);
      if (!newName?.trim()) return;
      await updateNameInCollection(itemsPath, docId, newName.trim());
      await loadCategories();
      return;
    }

    try {
      const id = await saveItem(clean, currentCategory.dataset.id, { stock: 0 });
      if (!id) return;

      const item = document.createElement("div");
      item.className = "item";
      item.textContent = clean;
      item.dataset.id = id;

      attachItemHandlerWithRetry(item, clean, currentUserId, currentCategory.dataset.id, id);

      currentCategory.querySelector(".children").appendChild(item);
      hideModal();
    } catch (err) {
      console.error("Failed to create item", err);
      alert("Failed to create item. See console for details.");
    }
  });

  deleteCatBtn?.addEventListener("click", async () => {
    if (!currentCategory || !currentUserId) return;
    const ok = confirm("Delete this category/subcategory? This will not delete child categories or items automatically.");
    if (!ok) return;
    const id = currentCategory.dataset.id;
    try {
      await deleteDoc(doc(db, "Shops", currentUserId, "categories", id));
      currentCategory.remove();
      hideModal();
    } catch (err) {
      console.error("Failed to delete category", err);
      alert("Failed to delete category. See console for details.");
    }
  });

  /* ------------------------------
     Utility: check if category is leaf (no child categories)
  -------------------------------*/
  async function isLeafCategory(uid, categoryId) {
    const q = query(collection(db, ...categoriesCollectionPath(uid)), where("parentId", "==", categoryId));
    const snap = await getDocs(q);
    return snap.empty;
  }

  // Ensure dynamic close buttons exist at startup
  injectOverlayCloseButtons();
});
