// smallSellingUnit.js - Selling Units UI Manager
// Version 3.0.0 - Firestore + Cloudinary + 2 required images

(function() {
  'use strict';

  // ========= Configuration =========
  const SELL_UNITS_COLLECTION = 'sellUnits'; // change to 'sellingUnits' if you prefer that name

  const IMAGES = {
    requiredCount: 2,
    maxSize: 5 * 1024 * 1024, // 5MB
    allowedTypes: ['image/jpeg', 'image/png', 'image/webp', 'image/jpg']
  };

  // ========= Integrations (injected once by your app/overlay) =========
  const INTEGRATIONS = { db: null, auth: null, FS: null, cloudName: null, uploadPreset: null };

  // Allow host app to inject Firebase + Cloudinary deps once
  window.sellingUnitsConfigure = function ({ db, auth, FS, cloudName, uploadPreset } = {}) {
    if (db) INTEGRATIONS.db = db;
    if (auth) INTEGRATIONS.auth = auth;
    if (FS) INTEGRATIONS.FS = FS; // { collection, addDoc, getDocs, deleteDoc, doc, updateDoc, serverTimestamp }
    if (cloudName) INTEGRATIONS.cloudName = cloudName;
    if (uploadPreset) INTEGRATIONS.uploadPreset = uploadPreset;
  };

  function requireConfigured() {
    if (!INTEGRATIONS.db || !INTEGRATIONS.FS) {
      throw new Error("SellingUnits: Firebase not configured. Call window.sellingUnitsConfigure(...) once.");
    }
    if (!INTEGRATIONS.cloudName || !INTEGRATIONS.uploadPreset) {
      throw new Error("SellingUnits: Cloudinary not configured. Pass cloudName and uploadPreset in sellingUnitsConfigure.");
    }
  }

  // ========= UI Config =========
  const CONFIG = {
    selectors: {
      itemDetail: '#item-detail, .item-detail',
      itemDetailHeader: '.item-detail-header',
      itemName: '#item-name'
    },
    button: {
      label: '📦 Selling Units',
      id: 'selling-units-btn',
      className: 'selling-units-btn'
    },
    modal: {
      id: 'selling-units-modal',
      className: 'selling-units-modal',
      backdropClass: 'selling-units-backdrop',
      title: '📦 Selling Units'
    }
  };

  // ========= State =========
  const state = {
    currentItemId: null,
    sellingUnits: new Map(), // itemId -> array of units [{id,name,conversionFactor,images,...}]
    observers: [],
    isModalOpen: false,
    loading: false
  };

  // ========= DOM cache =========
  const elements = {
    button: null,
    modal: null,
    backdrop: null
  };

  // ========= Utils =========
  function createElement(tag, attributes = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(attributes).forEach(([k, v]) => {
      if (k === 'className') el.className = v;
      else if (k === 'textContent') el.textContent = v;
      else if (k === 'innerHTML') el.innerHTML = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else el.setAttribute(k, v);
    });
    children.forEach(c => el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return el;
  }

  function clearElement(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function bytesToMB(b) {
    return (b / (1024 * 1024)).toFixed(1);
  }

  // ========= Styles =========
  function injectStyles() {
    const styleId = 'selling-units-styles';
    if (document.getElementById(styleId)) return;

    const styles = `
      .selling-units-btn {
        width: 100%;
        padding: 12px 16px;
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 12px;
        text-align: center;
        display: block;
        box-sizing: border-box;
      }
      .selling-units-btn:hover { background: #4338ca; transform: translateY(-1px); }
      .selling-units-btn:active { transform: translateY(0); }

      .selling-units-backdrop {
        position: fixed; inset: 0; background: rgba(0,0,0,0.5);
        z-index: 9999; display: flex; align-items: center; justify-content: center;
        animation: fadeIn 0.2s ease;
      }
      .selling-units-modal {
        background: white; border-radius: 8px; width: 90%; max-width: 520px;
        max-height: 85vh; overflow-y: auto; animation: slideIn 0.3s ease;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
      }
      .modal-header {
        padding: 20px 24px; border-bottom: 1px solid #e5e7eb;
        display: flex; justify-content: space-between; align-items: center;
      }
      .modal-title { margin: 0; font-size: 18px; font-weight: 600; color: #111827; }
      .close-btn { background: none; border: none; font-size: 24px; cursor: pointer; color: #6b7280; }
      .close-btn:hover { color: #111827; }
      .modal-body { padding: 24px; }

      .item-info { background: #f9fafb; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }
      .item-name { margin: 0 0 4px 0; font-weight: 500; color: #111827; }
      .base-unit { margin: 0; font-size: 14px; color: #6b7280; }

      .form-group { margin-bottom: 16px; }
      .form-label { display: block; margin-bottom: 6px; font-size: 14px; font-weight: 500; color: #374151; }
      .form-label.required::after { content: " *"; color: #ef4444; }
      .form-input { width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 14px; box-sizing: border-box; }
      .form-input:focus { outline: none; border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
      .help-text { font-size: 12px; color: #6b7280; margin-top: 4px; display:block; }

      .add-btn {
        width: 100%; padding: 10px 16px; background: #10b981; color: white;
        border: none; border-radius: 4px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.2s ease;
      }
      .add-btn:hover { background: #059669; }
      .add-btn[disabled] { opacity: .7; cursor: not-allowed; }

      .units-list { margin-top: 24px; }
      .list-title { font-size: 16px; font-weight: 600; margin: 0 0 12px 0; color: #111827; }
      .unit-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px; background: #f9fafb; border-radius: 4px; margin-bottom: 8px;
      }
      .unit-info { flex: 1; display:flex; gap:10px; align-items: center; }
      .unit-name { margin: 0 0 4px 0; font-weight: 500; color: #111827; }
      .unit-conversion { margin: 0; font-size: 14px; color: #6b7280; }
      .unit-thumb { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; border: 1px solid #e5e7eb; }

      .unit-remove {
        background: #ef4444; color: white; border: none; border-radius: 4px;
        width: 28px; height: 28px; cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center;
      }
      .unit-remove:hover { background: #dc2626; }

      .empty-state { text-align: center; padding: 20px; color: #6b7280; font-style: italic; }

      /* Image UI */
      .image-upload-section { margin-top: 8px; padding: 12px; background: #f9fafb; border-radius: 6px; }
      .image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .image-slot {
        border: 2px dashed #d1d5db; border-radius: 6px; height: 130px; background:white;
        display:flex; flex-direction:column; align-items:center; justify-content:center; cursor:pointer; position: relative; overflow:hidden;
      }
      .image-slot.has-image { border-style: solid; border-color: #10b981; }
      .image-icon { font-size: 22px; color:#9ca3af; margin-bottom: 6px; }
      .image-text { font-size: 12px; color:#6b7280; text-align:center; }
      .image-preview { width: 100%; height: 100%; object-fit: cover; display: block; }
      .image-remove {
        position: absolute; top: 6px; right: 6px; background: rgba(0,0,0,0.6);
        color: white; border: none; border-radius: 50%; width: 22px; height: 22px; font-size: 12px; cursor: pointer;
        display:flex; align-items:center; justify-content:center;
      }

      @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
      @keyframes slideIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
    `;
    document.head.appendChild(createElement('style', { id: styleId }, [styles]));
  }

  // ========= Context helpers =========
  function resolveItemContext() {
    const el = document.querySelector('#item-detail');
    if (!el) return null;
    const shopId = el.dataset.shopId;
    const categoryId = el.dataset.categoryId;
    const itemId = el.dataset.itemId;
    const itemName = el.dataset.itemName || document.querySelector(CONFIG.selectors.itemName)?.textContent || "";
    const baseUnit = el.dataset.baseUnit || "unit";
    if (!shopId || !categoryId || !itemId) return null;
    return { shopId, categoryId, itemId, itemName, baseUnit };
  }

  function getCurrentItemId() {
    return resolveItemContext()?.itemId || null;
  }

  function getBaseUnit() {
    return resolveItemContext()?.baseUnit || "Unit";
  }

  function getItemName() {
    return resolveItemContext()?.itemName || "Unknown Item";
  }

  // ========= Cloudinary =========
  function validateFile(file) {
    if (!IMAGES.allowedTypes.includes(file.type)) {
      throw new Error(`Unsupported type: ${file.type}`);
    }
    if (file.size > IMAGES.maxSize) {
      throw new Error(`Max size is ${bytesToMB(IMAGES.maxSize)}MB. Got ${bytesToMB(file.size)}MB`);
    }
  }

  async function uploadToCloudinary(file) {
    requireConfigured();
    validateFile(file);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("upload_preset", INTEGRATIONS.uploadPreset);

    const res = await fetch(`https://api.cloudinary.com/v1_1/${INTEGRATIONS.cloudName}/image/upload`, {
      method: "POST",
      body: fd
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    return data;
  }

  function toImageMeta(cld) {
    return {
      url: cld.secure_url,
      thumb: cld.secure_url.replace("/upload/", "/upload/c_fill,w_300,h_300,q_auto,f_auto/"),
      publicId: cld.public_id,
      width: cld.width,
      height: cld.height,
      bytes: cld.bytes,
      format: cld.format
    };
  }

  // ========= Firestore helpers =========
  function sellingUnitsColRef(ctx) {
    const { collection } = INTEGRATIONS.FS;
    return collection(
      INTEGRATIONS.db,
      "Shops", ctx.shopId,
      "categories", ctx.categoryId,
      "items", ctx.itemId,
      SELL_UNITS_COLLECTION
    );
  }

  async function fetchSellingUnits(ctx) {
    requireConfigured();
    const { getDocs } = INTEGRATIONS.FS;
    const snap = await getDocs(sellingUnitsColRef(ctx));
    return snap.docs.map(d => ({ id: d.id, ...d.data() }));
  }

  async function createSellingUnit(ctx, { name, conversionFactor, files }) {
    requireConfigured();
    const { addDoc, serverTimestamp } = INTEGRATIONS.FS;
    const user = INTEGRATIONS.auth?.currentUser || null;

    // Upload exactly two images
    if (!files || files.length < IMAGES.requiredCount) {
      throw new Error(`Please add ${IMAGES.requiredCount} images`);
    }
    const [f1, f2] = files;
    const [c1, c2] = await Promise.all([uploadToCloudinary(f1), uploadToCloudinary(f2)]);
    const images = [toImageMeta(c1), toImageMeta(c2)];

    const data = {
      name,
      // store both names for compatibility with older/newer code
      conversionFactor: Number(conversionFactor),
      conversion: Number(conversionFactor),
      baseUnit: ctx.baseUnit,
      itemName: ctx.itemName,
      images,
      createdAt: serverTimestamp(),
      createdBy: user ? { uid: user.uid, email: user.email || null, name: user.displayName || null } : null
    };

    const ref = await addDoc(sellingUnitsColRef(ctx), data);
    return { id: ref.id, ...data };
  }

  async function deleteSellingUnitDoc(ctx, id) {
    requireConfigured();
    const { deleteDoc, doc } = INTEGRATIONS.FS;
    const ref = doc(
      INTEGRATIONS.db,
      "Shops", ctx.shopId,
      "categories", ctx.categoryId,
      "items", ctx.itemId,
      SELL_UNITS_COLLECTION, id
    );
    await deleteDoc(ref);
  }

  // ========= Button injection =========
  function createButton() {
    return createElement('button', {
      id: CONFIG.button.id,
      className: CONFIG.button.className,
      textContent: CONFIG.button.label,
      type: 'button'
    });
  }

  function injectButton(itemDetail) {
    let btn = itemDetail.querySelector(`#${CONFIG.button.id}`);
    if (btn) {
      elements.button = btn;
      if (!btn.dataset.suBound) {
        btn.addEventListener('click', handleButtonClick);
        btn.dataset.suBound = "1";
      }
      return;
    }

    const header = itemDetail.querySelector(CONFIG.selectors.itemDetailHeader);
    if (!header) {
      console.warn('SellingUnits: item detail header not found');
      return;
    }
    btn = createButton();
    elements.button = btn;
    header.parentNode.insertBefore(btn, header.nextSibling);
    btn.addEventListener('click', handleButtonClick);
    btn.dataset.suBound = "1";
  }

  function handleButtonClick(e) {
    e.stopPropagation();
    e.preventDefault();
    openModal();
  }

  // ========= Modal =========
  function createModal() {
    elements.backdrop = createElement('div', { className: CONFIG.modal.backdropClass });
    elements.modal = createElement('div', { className: CONFIG.modal.className, id: CONFIG.modal.id });

    elements.backdrop.addEventListener('click', (e) => {
      if (e.target === elements.backdrop) closeModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && state.isModalOpen) closeModal();
    });
    return elements.modal;
  }

  async function openModal() {
    if (state.isModalOpen) return;

    const ctx = resolveItemContext();
    if (!ctx) {
      alert("Missing item context. Ensure #item-detail has data-shop-id, data-category-id, data-item-id.");
      return;
    }

    state.isModalOpen = true;
    state.currentItemId = ctx.itemId;

    if (!elements.modal) createModal();
    clearElement(elements.modal);

    const itemName = getItemName();
    const baseUnit = getBaseUnit();
    const currentUnits = state.sellingUnits.get(state.currentItemId) || [];

    const modalContent = buildModalContent(itemName, baseUnit, currentUnits);
    elements.modal.appendChild(modalContent);
    elements.backdrop.appendChild(elements.modal);
    document.body.appendChild(elements.backdrop);
    document.body.style.overflow = 'hidden';

    // Fetch latest from Firestore and refresh
    try {
      state.loading = true;
      const units = await fetchSellingUnits(ctx);
      state.sellingUnits.set(state.currentItemId, units);
      refreshModal();
    } catch (err) {
      console.error("SellingUnits: failed to fetch units", err);
    } finally {
      state.loading = false;
    }
  }

  function closeModal() {
    if (!state.isModalOpen || !elements.backdrop) return;
    state.isModalOpen = false;
    if (elements.backdrop.parentNode) elements.backdrop.parentNode.removeChild(elements.backdrop);
    document.body.style.overflow = '';
  }

  function buildModalContent(itemName, baseUnit, sellingUnits) {
    const fragment = document.createDocumentFragment();

    const header = createElement('div', { className: 'modal-header' }, [
      createElement('h2', { className: 'modal-title', textContent: CONFIG.modal.title }),
      createElement('button', { className: 'close-btn', textContent: '×' })
    ]);
    header.querySelector('.close-btn').addEventListener('click', closeModal);
    fragment.appendChild(header);

    const body = createElement('div', { className: 'modal-body' });

    const itemInfo = createElement('div', { className: 'item-info' }, [
      createElement('h3', { className: 'item-name', textContent: itemName }),
      createElement('p', { className: 'base-unit', textContent: `Base unit: ${baseUnit}` })
    ]);
    body.appendChild(itemInfo);

    const form = createForm(baseUnit);
    body.appendChild(form);

    const list = createUnitsList(sellingUnits, baseUnit);
    body.appendChild(list);

    fragment.appendChild(body);
    return fragment;
  }

  function createImageSlot(label, onFileSelected) {
    const slot = createElement('div', { className: 'image-slot' }, [
      createElement('div', { className: 'image-icon', textContent: '📷' }),
      createElement('div', { className: 'image-text', textContent: label })
    ]);

    const input = createElement('input', {
      type: 'file', accept: IMAGES.allowedTypes.join(','), style: { display: 'none' }
    });

    // Prefer camera on mobile
    input.setAttribute('capture', 'environment');

    slot.addEventListener('click', () => input.click());
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        validateFile(file);
      } catch (e) {
        alert(e.message);
        input.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        slot.classList.add('has-image');
        slot.innerHTML = '';
        const img = createElement('img', { className: 'image-preview', src: reader.result });
        const remove = createElement('button', { className: 'image-remove', type: 'button', textContent: '×' });
        remove.addEventListener('click', (ev) => {
          ev.stopPropagation();
          input.value = '';
          slot.classList.remove('has-image');
          slot.innerHTML = '';
          slot.appendChild(createElement('div', { className: 'image-icon', textContent: '📷' }));
          slot.appendChild(createElement('div', { className: 'image-text', textContent: label }));
          onFileSelected(null);
        });
        slot.appendChild(img);
        slot.appendChild(remove);
        onFileSelected(file);
      };
      reader.readAsDataURL(file);
    });

    return { slot, input };
  }

  function createForm(baseUnit) {
    const form = createElement('form', { className: 'selling-unit-form' });

    const nameGroup = createElement('div', { className: 'form-group' }, [
      createElement('label', { className: 'form-label required', textContent: 'Selling Unit Name' }),
      createElement('input', { className: 'form-input', type: 'text', placeholder: 'e.g., Piece, Packet, Box', required: true }),
      createElement('small', { className: 'help-text', textContent: 'What customers will see (e.g., “Piece”).' })
    ]);
    const conversionGroup = createElement('div', { className: 'form-group' }, [
      createElement('label', { className: 'form-label required', textContent: 'Conversion Factor' }),
      createElement('input', { className: 'form-input', type: 'number', min: '1', step: '1', placeholder: `e.g., 12 for 12 in a ${baseUnit}`, required: true }),
      createElement('small', { className: 'help-text', textContent: `How many of this unit are in 1 ${baseUnit}?` })
    ]);

    // Images section (2 required)
    const imageSection = createElement('div', { className: 'image-upload-section' }, [
      createElement('div', { className: 'form-label required', textContent: 'Unit Images (2 required)' }),
      createElement('small', { className: 'help-text', textContent: `Max ${bytesToMB(IMAGES.maxSize)}MB • Formats: jpg, png, webp` }),
      createElement('div', { className: 'image-grid' })
    ]);
    const grid = imageSection.querySelector('.image-grid');

    const selectedFiles = [null, null];

    const s1 = createImageSlot('Click to add Image 1', (f) => { selectedFiles[0] = f; });
    const s2 = createImageSlot('Click to add Image 2', (f) => { selectedFiles[1] = f; });
    grid.appendChild(s1.slot);
    grid.appendChild(s2.slot);

    const addButton = createElement('button', { className: 'add-btn', type: 'submit', textContent: '➕ Add Selling Unit' });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const ctx = resolveItemContext();
      if (!ctx) { alert("Missing item context"); return; }

      const nameInput = nameGroup.querySelector('input');
      const convInput = conversionGroup.querySelector('input');

      const unitName = nameInput.value.trim();
      const conversionFactor = parseFloat(convInput.value);

      if (!unitName) { nameInput.focus(); return; }
      if (!conversionFactor || conversionFactor <= 0 || !isFinite(conversionFactor)) { convInput.focus(); return; }

      // Validate two images
      if (!selectedFiles[0] || !selectedFiles[1]) {
        alert(`Please add ${IMAGES.requiredCount} images`);
        return;
      }

      try {
        addButton.disabled = true;
        // Requested: turn Add → Save when clicked
        addButton.textContent = '💾 Save...';

        const created = await createSellingUnit(ctx, {
          name: unitName,
          conversionFactor,
          files: selectedFiles
        });

        const arr = state.sellingUnits.get(ctx.itemId) || [];
        arr.push(created);
        state.sellingUnits.set(ctx.itemId, arr);
        refreshModal();

        // Reset form
        nameInput.value = '';
        convInput.value = '';
        // reset image slots
        s1.input.value = '';
        s2.input.value = '';
        selectedFiles[0] = selectedFiles[1] = null;
        s1.slot.classList.remove('has-image');
        s1.slot.innerHTML = '';
        s1.slot.appendChild(createElement('div', { className: 'image-icon', textContent: '📷' }));
        s1.slot.appendChild(createElement('div', { className: 'image-text', textContent: 'Click to add Image 1' }));
        s2.slot.classList.remove('has-image');
        s2.slot.innerHTML = '';
        s2.slot.appendChild(createElement('div', { className: 'image-icon', textContent: '📷' }));
        s2.slot.appendChild(createElement('div', { className: 'image-text', textContent: 'Click to add Image 2' }));

        nameInput.focus();
      } catch (err) {
        console.error("SellingUnits: failed to save", err);
        alert("Failed to save selling unit");
      } finally {
        addButton.disabled = false;
        addButton.textContent = '➕ Add Selling Unit';
      }
    });

    form.appendChild(nameGroup);
    form.appendChild(conversionGroup);
    form.appendChild(imageSection);
    form.appendChild(addButton);
    return form;
  }

  function createUnitsList(sellingUnits, baseUnit) {
    const container = createElement('div', { className: 'units-list' });
    container.appendChild(createElement('h3', { className: 'list-title', textContent: 'Selling Units' }));

    if (!sellingUnits || sellingUnits.length === 0) {
      container.appendChild(createElement('p', { className: 'empty-state', textContent: 'No selling units added yet' }));
      return container;
    }

    sellingUnits.forEach((unit, index) => {
      const img = Array.isArray(unit.images) && unit.images[0] ? unit.images[0] : null;
      const thumbSrc = (img && (img.thumb || img.url)) || '';

      const unitItem = createElement('div', { className: 'unit-item' }, [
        createElement('div', { className: 'unit-info' }, [
          thumbSrc ? createElement('img', { className: 'unit-thumb', src: thumbSrc, alt: unit.name }) : createElement('div', { className: 'unit-thumb', style: { background: '#eef2ff', display:'flex', alignItems:'center', justifyContent:'center', color:'#6366f1', fontSize:'12px' } }, ['📷']),
          createElement('div', {}, [
            createElement('h4', { className: 'unit-name', textContent: unit.name }),
            createElement('p', {
              className: 'unit-conversion',
              textContent: `1 ${unit.name} = ${unit.conversionFactor ?? unit.conversion} ${baseUnit}`
            })
          ])
        ]),
        createElement('button', { className: 'unit-remove', 'data-index': String(index), title: 'Remove' }, ['×'])
      ]);

      unitItem.querySelector('.unit-remove').addEventListener('click', () => removeSellingUnit(index));
      container.appendChild(unitItem);
    });

    return container;
  }

  // ========= Data ops for list =========
  async function removeSellingUnit(index) {
    const ctx = resolveItemContext();
    if (!ctx) return;

    const arr = state.sellingUnits.get(ctx.itemId) || [];
    const unit = arr[index];
    if (!unit) return;

    if (!confirm(`Remove selling unit "${unit.name}"?`)) return;

    try {
      if (unit.id) await deleteSellingUnitDoc(ctx, unit.id);
      arr.splice(index, 1);
      state.sellingUnits.set(ctx.itemId, arr);
      refreshModal();
    } catch (err) {
      console.error("SellingUnits: failed to delete", err);
      alert("Failed to delete selling unit");
    }
  }

  function refreshModal() {
    if (!state.isModalOpen || !state.currentItemId) return;
    clearElement(elements.modal);
    const itemName = getItemName();
    const baseUnit = getBaseUnit();
    const currentUnits = state.sellingUnits.get(state.currentItemId) || [];
    elements.modal.appendChild(buildModalContent(itemName, baseUnit, currentUnits));
  }

  // ========= Observers / Init =========
  function setupObservers() {
    document.addEventListener("item-detail:opened", () => {
      const existing = document.querySelector(CONFIG.selectors.itemDetail);
      if (existing) injectButton(existing);
    });

    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType !== Node.ELEMENT_NODE) continue;
          const itemDetail = node.matches?.(CONFIG.selectors.itemDetail)
            ? node : node.querySelector?.(CONFIG.selectors.itemDetail);
          if (itemDetail) injectButton(itemDetail);
        }
      }
      const existing = document.querySelector(CONFIG.selectors.itemDetail);
      if (existing && !existing.querySelector(`#${CONFIG.button.id}`)) injectButton(existing);
    });

    observer.observe(document.body, { childList: true, subtree: true });
    state.observers.push(observer);
  }

  function cleanup() {
    state.observers.forEach(o => o.disconnect());
    state.observers = [];
    if (elements.button) {
      const clone = elements.button.cloneNode(true);
      elements.button.parentNode?.replaceChild(clone, elements.button);
      elements.button = null;
    }
    closeModal();
    const styles = document.getElementById('selling-units-styles');
    if (styles) styles.parentNode?.removeChild(styles);
  }

  function init() {
    try {
      cleanup();
      injectStyles();

      const existing = document.querySelector(CONFIG.selectors.itemDetail);
      if (existing) injectButton(existing);

      setupObservers();
      console.log('Selling Units manager initialized (Firestore + Cloudinary, 2 required images)');
    } catch (err) {
      console.error('Failed to initialize Selling Units manager:', err);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.sellingUnitsCleanup = cleanup;

})();