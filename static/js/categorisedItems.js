// categorisedItems.js (UPDATED: staff support + RBAC + audit logs)
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
  where,
  serverTimestamp
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

  const itemDetail = document.getElementById("item-detail");

  // State
  let currentCategory = null;
  let currentNodeData = null;

  let currentShopId = null;      // ✅ shop context (owner uid)
  let currentAuthUid = null;     // ✅ auth user uid (owner/staff)
  let currentActor = null;       // ✅ who is performing actions

  // ------------------------------
  // Session helpers (NEW)
  // ------------------------------
  function safeParse(json) {
    try { return JSON.parse(json); } catch { return null; }
  }

  function getSessionType() {
    return localStorage.getItem("sessionType") || "owner";
  }

  function getAccessLevel() {
    const t = getSessionType();
    if (t === "staff") {
      const ctx = safeParse(localStorage.getItem("staffContext")) || {};
      return Number(ctx.accessLevel ?? 1);
    }
    return 4; // owner full access
  }

  function canManageStock() {
    // adjust if your rules differ
    return getAccessLevel() >= 2;
  }

  function resolveShopId() {
    const shopId = localStorage.getItem("activeShopId");
    return shopId || null;
  }

  function resolveActor(authUser) {
    const t = getSessionType();
    if (t === "staff") {
      const ctx = safeParse(localStorage.getItem("staffContext")) || {};
      return {
        type: "staff",
        authUid: authUser.uid,
        staffId: ctx.staffId || null,
        name: ctx.name || authUser.displayName || "",
        email: ctx.email || authUser.email || "",
        roleName: ctx.roleName || "",
        accessLevel: ctx.accessLevel ?? null,
        shopId: ctx.shopId || null
      };
    }
    return {
      type: "owner",
      authUid: authUser.uid,
      name: authUser.displayName || "",
      email: authUser.email || "",
      accessLevel: 4
    };
  }

  async function writeAuditLog(action, entityType, entityId, extra = {}) {
    // Best-effort logging: if rules block it, we do not break the main flow
    try {
      if (!currentShopId) return;

      await addDoc(collection(db, "Shops", currentShopId, "auditLogs"), {
        action,                 // e.g. "create", "update", "delete"
        entityType,             // "category" | "item"
        entityId: entityId || null,
        shopId: currentShopId,

        performedBy: currentActor || null,
        performedByDisplay:
          currentActor?.name || currentActor?.email || currentActor?.authUid || "unknown",

        timestamp: serverTimestamp(),
        ...extra
      });
    } catch (e) {
      console.warn("Audit log failed (non-blocking):", e);
    }
  }

  function requireManageStockOrBlock() {
    if (!canManageStock()) {
      alert("Access denied: You don’t have permission to manage stock.");
      return false;
    }
    return true;
  }

  /* ------------------------------
     Firestore path helpers
  -------------------------------*/
  function categoriesCollectionPath(shopId) {
    return ["Shops", shopId, "categories"];
  }
  function itemsCollectionPath(shopId, categoryId) {
    return ["Shops", shopId, "categories", categoryId, "items"];
  }

  /* ------------------------------
     Attach item handler (robust)
  -------------------------------*/
  function attachItemHandlerWithRetry(el, name, shopId, categoryId, itemId) {
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
          window.attachItemDetailHandler(el, name, shopId, categoryId, itemId);
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

    // RBAC: hide mutation actions if cannot manage stock
    const allow = canManageStock();
    addSubBtn.style.display = allow ? "inline-block" : "none";
    addItemBtn.style.display = allow ? "inline-block" : "none";
    deleteCatBtn.style.display = allow ? "inline-block" : "none";

    const children = currentCategory.querySelector(".children")?.children || [];
    const hasSubcategories = Array.from(children).some(c => c.classList.contains("category-item"));
    const hasItems = Array.from(children).some(c => c.classList.contains("item"));

    // original logic
    if (allow) {
      addSubBtn.style.display = hasItems ? "none" : "inline-block";
      addItemBtn.style.display = hasSubcategories ? "none" : "inline-block";
    }
  }

  /* ------------------------------
     Overlay helpers
  -------------------------------*/
  function showCategoriesOverlay() {
    overlay.classList.remove("hidden");
    overlayContent.classList.remove("hidden");
    if (itemDetail) {
      itemDetail.classList.add("hidden");
      itemDetail.setAttribute("aria-hidden", "true");
    }
    injectCategoriesCloseButton();
  }

  function closeCategoriesOverlay() {
    hideOverlayCompletely();
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

  function injectCategoriesCloseButton() {
    if (overlay && !document.getElementById("categories-close-btn")) {
      const closeBtn = document.createElement("span");
      closeBtn.id = "categories-close-btn";
      closeBtn.className = "close-x";
      closeBtn.setAttribute("role", "button");
      closeBtn.setAttribute("aria-label", "Close categories and go back to dashboard");
      closeBtn.innerHTML = "&times;";
      closeBtn.style.position = "absolute";
      closeBtn.style.top = "10px";
      closeBtn.style.right = "15px";
      closeBtn.style.fontSize = "24px";
      closeBtn.style.cursor = "pointer";
      closeBtn.style.zIndex = "1001";
      overlay.appendChild(closeBtn);

      closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        closeCategoriesOverlay();
      });
    }
  }

  /* ------------------------------
     Load categories and items
  -------------------------------*/
  async function loadCategories() {
    if (!currentShopId) return;
    categoriesList.innerHTML = "";

    const catSnap = await getDocs(collection(db, ...categoriesCollectionPath(currentShopId)));
    const map = {};

    catSnap.forEach(d => {
      const data = d.data();
      map[d.id] = {
        node: createCategoryNode(data.name, d.id),
        parentId: data.parentId
      };
    });

    Object.values(map).forEach(({ node, parentId }) => {
      if (parentId && map[parentId]) {
        map[parentId].node.querySelector(".children").appendChild(node);
      } else {
        categoriesList.appendChild(node);
      }
    });

    for (const catId of Object.keys(map)) {
      const itemsSnap = await getDocs(collection(db, ...itemsCollectionPath(currentShopId, catId)));
      itemsSnap.forEach(d => {
        const data = d.data();
        const parent = map[catId]?.node;
        if (!parent) return;

        const item = document.createElement("div");
        item.className = "item";
        item.textContent = data.name;
        item.dataset.id = d.id;

        attachItemHandlerWithRetry(item, data.name, currentShopId, catId, d.id);
        parent.querySelector(".children").appendChild(item);
      });
    }
  }

  window.reloadShopCategories = loadCategories;

  /* ------------------------------
     Auth watcher (FIXED)
  -------------------------------*/
  const auth = getAuth();
  auth.onAuthStateChanged(user => {
    if (!user) {
      currentShopId = null;
      currentAuthUid = null;
      currentActor = null;
      if (categoriesList) categoriesList.innerHTML = "";
      return;
    }

    currentAuthUid = user.uid;
    currentShopId = resolveShopId();      // ✅ critical: shop context
    currentActor = resolveActor(user);    // ✅ who is acting

    if (!currentShopId) {
      console.warn("activeShopId missing. Redirecting to home.");
      window.location.href = "/";
      return;
    }

    loadCategories().catch(err => console.error("Failed to load categories:", err));
  });

  /* ------------------------------
     Event bindings
  -------------------------------*/
  manageStockBtn?.addEventListener("click", () => {
    // You may allow staff to view but not edit; up to you.
    // If you want staff L1 to not even open overlay, enforce here:
    if (!canManageStock()) {
      alert("Access denied: You don’t have permission to manage stock.");
      return;
    }
    showCategoriesOverlay();
  });

  overlay?.addEventListener("click", (e) => {
    if (e.target === overlay) hideOverlayCompletely();
  });

  /* ------------------------------
     CRUD helpers (UPDATED WITH TRACKING)
  -------------------------------*/
  async function saveCategory(name, parentId = null) {
    if (!currentShopId) return null;
    let ancestors = [];
    let fullPath = name;

    if (parentId) {
      const parentRef = doc(db, "Shops", currentShopId, "categories", parentId);
      const parentSnap = await getDoc(parentRef);
      if (!parentSnap.exists()) throw new Error("Parent category not found");

      const parent = parentSnap.data();
      ancestors = Array.isArray(parent.ancestors) ? [...parent.ancestors] : [];
      ancestors.push({ id: parentId, name: parent.name });
      fullPath = ancestors.map(a => a.name).concat(name).join(" > ");
    }

    const ref = await addDoc(collection(db, ...categoriesCollectionPath(currentShopId)), {
      name,
      parentId,
      ancestors,
      fullPath,
      createdAt: Date.now(),
      createdBy: currentActor || null
    });

    await writeAuditLog("create", "category", ref.id, { name, parentId });
    return ref.id;
  }

  async function saveItem(name, parentId, itemData = {}) {
    if (!currentShopId) return null;

    const catRef = doc(db, "Shops", currentShopId, "categories", parentId);
    const catSnap = await getDoc(catRef);
    if (!catSnap.exists()) throw new Error("Category not found");

    const cat = catSnap.data();
    const ancestors = Array.isArray(cat.ancestors) ? [...cat.ancestors] : [];
    ancestors.push({ id: parentId, name: cat.name });
    const fullPath = ancestors.map(a => a.name).concat(name).join(" > ");

    const ref = await addDoc(collection(db, ...itemsCollectionPath(currentShopId, parentId)), {
      name,
      categoryId: parentId,
      ancestors,
      fullPath,
      ...itemData,
      createdAt: Date.now(),
      createdBy: currentActor || null
    });

    await writeAuditLog("create", "item", ref.id, { name, categoryId: parentId });
    return ref.id;
  }

  async function nameExistsInCollection(collectionPath, name) {
    const colRef = collection(db, ...collectionPath);
    const snap = await getDocs(colRef);
    const key = name.trim().toLowerCase();
    const existingDoc = snap.docs.find(d => (d.data().name || "").toLowerCase() === key);
    return existingDoc
      ? { exists: true, docId: existingDoc.id, data: existingDoc.data() }
      : { exists: false };
  }

  async function updateNameInCollection(collectionPath, docId, newName) {
    await updateDoc(doc(db, ...collectionPath, docId), {
      name: newName,
      updatedAt: Date.now(),
      updatedBy: currentActor || null
    });

    await writeAuditLog("update", collectionPath[collectionPath.length - 1] === "categories" ? "category" : "item", docId, {
      field: "name",
      newName
    });

    if (collectionPath.length >= 3 && collectionPath[collectionPath.length - 1] === "categories") {
      await rebuildAllCategoryPaths(currentShopId);
    }
  }

  async function rebuildAllCategoryPaths(shopId) {
    const catSnap = await getDocs(collection(db, ...categoriesCollectionPath(shopId)));
    const map = {};
    catSnap.forEach(d => (map[d.id] = { id: d.id, ...d.data() }));

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
      await updateDoc(doc(db, "Shops", shopId, "categories", id), {
        ancestors,
        fullPath,
        updatedAt: Date.now(),
        updatedBy: currentActor || null
      });
    }

    for (const id of Object.keys(map)) {
      const cat = map[id];
      const catAncestors = Array.isArray(cat.ancestors) ? [...cat.ancestors] : [];
      const itemsSnap = await getDocs(collection(db, ...itemsCollectionPath(shopId, id)));
      for (const itemDoc of itemsSnap.docs) {
        const item = itemDoc.data();
        const itemAncestors = [...catAncestors, { id, name: cat.name }];
        const itemFullPath = itemAncestors.map(a => a.name).concat(item.name).join(" > ");
        await updateDoc(doc(db, "Shops", shopId, "categories", id, "items", itemDoc.id), {
          ancestors: itemAncestors,
          fullPath: itemFullPath,
          updatedAt: Date.now(),
          updatedBy: currentActor || null
        });
      }
    }
  }

  /* ------------------------------
     Creation and deletion bindings (UPDATED WITH RBAC)
  -------------------------------*/
  categoriesBtn?.addEventListener("click", async () => {
    if (!requireManageStockOrBlock()) return;
    if (!currentShopId) return;

    const name = prompt("Enter category name:");
    if (!name?.trim()) return;
    const clean = name.trim();

    const { exists, docId } = await nameExistsInCollection(categoriesCollectionPath(currentShopId), clean);
    if (exists) {
      const confirmEdit = confirm(`Category "${clean}" already exists. Do you want to rename it?`);
      if (!confirmEdit) return;
      const newName = prompt("Enter new name for category:", clean);
      if (!newName?.trim()) return;
      await updateNameInCollection(categoriesCollectionPath(currentShopId), docId, newName.trim());
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
    if (!requireManageStockOrBlock()) return;
    if (!currentCategory || !currentShopId) return;

    const name = prompt("Enter subcategory name:");
    if (!name?.trim()) return;
    const clean = name.trim();

    const { exists, docId } = await nameExistsInCollection(categoriesCollectionPath(currentShopId), clean);
    if (exists) {
      const confirmEdit = confirm(`Subcategory "${clean}" already exists. Do you want to rename it?`);
      if (!confirmEdit) return;
      const newName = prompt("Enter new name for subcategory:", clean);
      if (!newName?.trim()) return;
      await updateNameInCollection(categoriesCollectionPath(currentShopId), docId, newName.trim());
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
    if (!requireManageStockOrBlock()) return;
    if (!currentCategory || !currentShopId) return;

    const isLeaf = await isLeafCategory(currentShopId, currentCategory.dataset.id);
    if (!isLeaf) {
      alert("This category has subcategories. Add items only to a leaf category.");
      return;
    }

    const name = prompt("Enter item name:");
    if (!name?.trim()) return;
    const clean = name.trim();

    const itemsPath = itemsCollectionPath(currentShopId, currentCategory.dataset.id);
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

      attachItemHandlerWithRetry(item, clean, currentShopId, currentCategory.dataset.id, id);
      currentCategory.querySelector(".children").appendChild(item);
      hideModal();
    } catch (err) {
      console.error("Failed to create item", err);
      alert("Failed to create item. See console for details.");
    }
  });

  deleteCatBtn?.addEventListener("click", async () => {
    if (!requireManageStockOrBlock()) return;
    if (!currentCategory || !currentShopId) return;

    const ok = confirm("Delete this category/subcategory? This will not delete child categories or items automatically.");
    if (!ok) return;

    const id = currentCategory.dataset.id;

    try {
      await writeAuditLog("delete", "category", id, { name: currentNodeData?.name || null });
      await deleteDoc(doc(db, "Shops", currentShopId, "categories", id));
      currentCategory.remove();
      hideModal();
    } catch (err) {
      console.error("Failed to delete category", err);
      alert("Failed to delete category. See console for details.");
    }
  });

  async function isLeafCategory(shopId, categoryId) {
    const q = query(collection(db, ...categoriesCollectionPath(shopId)), where("parentId", "==", categoryId));
    const snap = await getDocs(q);
    return snap.empty;
  }
});